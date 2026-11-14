#include "inference.hpp"
#include "logger.hpp"
#include <csignal>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <poll.h>
#include <random>
#include <termios.h>
#include <sys/ioctl.h>
#include <unistd.h>
namespace {
volatile std::sig_atomic_t stopping=0;
void stop_signal(int) { stopping=1; }
struct Serial {
    int fd=-1;
    explicit Serial(const std::string& path) {
        fd=open(path.c_str(),O_RDWR|O_NOCTTY|O_NONBLOCK);if(fd<0) throw std::runtime_error("cannot open serial: "+path);
        if(ioctl(fd,TIOCEXCL)) { close(fd);fd=-1;throw std::runtime_error("serial already owned"); }
        termios t{};
        if(tcgetattr(fd,&t)) { close(fd);fd=-1;throw std::runtime_error("tcgetattr"); }
        cfmakeraw(&t);cfsetispeed(&t,B1152000);cfsetospeed(&t,B1152000);
        t.c_cflag=(t.c_cflag&~(PARENB|CSTOPB|CSIZE|CRTSCTS))|CS8|CLOCAL|CREAD;
        if(tcsetattr(fd,TCSANOW,&t)) { close(fd);fd=-1;throw std::runtime_error("tcsetattr"); }
        tcflush(fd,TCIFLUSH);
    }
    ~Serial() { if(fd>=0) close(fd); }
    void send(const uint8_t* p,size_t n) {
        uint64_t end=jrl::monotonic_us()+2000u;size_t sent=0;
        while(sent<n) {
            ssize_t k=write(fd,p+sent,n-sent);
            if(k>0) sent+=size_t(k);
            else if(k<0&&errno!=EAGAIN&&errno!=EINTR) throw std::runtime_error("serial write failed");
            if(sent<n) { if(jrl::monotonic_us()>=end) throw std::runtime_error("serial write deadline");pollfd d{fd,POLLOUT,0};poll(&d,1,1); }
        }
    }
};
using Args=std::map<std::string,std::string>;
std::string need(const Args& a,const char* key) { auto it=a.find(key);if(it==a.end()) throw std::runtime_error(std::string("missing ")+key);return it->second; }
void vectors(jrl::Inference& net,const std::string& input,const std::string& output) {
    if(std::filesystem::exists(output)) throw std::runtime_error("refuse to overwrite output");
    std::ifstream in(input,std::ios::binary);std::ofstream out(output,std::ios::binary);if(!in||!out) throw std::runtime_error("vector files");
    jrl::Input x;size_t count=0;uint64_t start=jrl::monotonic_us();std::vector<uint64_t> timings;
    while(in.read(reinterpret_cast<char*>(x.data()),sizeof(x))) { auto begin=jrl::monotonic_us();auto y=net.run(x);timings.push_back(jrl::monotonic_us()-begin);out.write(reinterpret_cast<char*>(y.data()),sizeof(y));count++; }
    if(in.gcount()!=0||!in.eof()||!out||!count) throw std::runtime_error("incomplete vectors");
    std::sort(timings.begin(),timings.end());
    std::cout<<"vectors="<<count<<" total_us="<<jrl::monotonic_us()-start<<" inference_p95_us="<<timings[(timings.size()-1)*95/100]<<" inference_max_us="<<timings.back()<<'\n';
}
void replay(jrl::Inference& net,const std::string& input,const std::string& output) {
    if(std::filesystem::exists(output)) throw std::runtime_error("refuse to overwrite replay");
    std::ifstream in(input,std::ios::binary);std::ofstream out(output);if(!in||!out) throw std::runtime_error("replay files");
    out<<"sequence,max_abs_error,match\n";size_t count=0;bool all=true;uint8_t h[16];
    while(in.read(reinterpret_cast<char*>(h),16)) {
        uint32_t n=jrl_u32(h+4);if(n>4096) throw std::runtime_error("bad log record size");std::vector<uint8_t> b(n);
        if(!in.read(reinterpret_cast<char*>(b.data()),n)) throw std::runtime_error("truncated log payload");
        if(jrl_u32(h)!=3) continue;
        if(n!=640) throw std::runtime_error("bad inference record");
        jrl::Input x;for(unsigned i=0;i<150;i++) x[i]=jrl_f32(b.data()+16+4*i);
        jrl::Action y{};if(!jrl_u32(b.data()+12)) y=net.run(x);
        bool ok=true;float error=0;
        for(unsigned i=0;i<6;i++) { float actual=jrl::clip(y[i],100.f),expected=jrl_f32(b.data()+616+4*i),e=std::abs(actual-expected);error=std::max(error,e);ok&=e<=1e-4f+1e-4f*std::abs(expected); }
        out<<jrl_u32(b.data()+4)<<','<<error<<','<<ok<<'\n';all&=ok;count++;
    }
    if(in.gcount()!=0||!in.eof()||!out||!count||!all) throw std::runtime_error("replay incomplete or mismatch");
}
void live(const Args& a,bool control) {
    const std::string directory=need(a,"--log-dir"),port=need(a,"--port");
    std::unique_ptr<jrl::Inference> net;
    if(control) net=std::make_unique<jrl::Inference>(need(a,"--engine"));
    jrl::Logger log(directory);
    std::ofstream meta(directory+"/meta.txt");
    meta<<"protocol=2\ncontract="<<JRL_CONTRACT<<"\nmode="<<(control?"control":"record")<<"\nport="<<port
        <<"\nexecutable_sha256="<<jrl::sha256("/proc/self/exe")<<"\nconfig_sha256="<<JRL_CONFIG_SHA<<"\nconfig=17679_main_pitch-2_history-v2_ros-standing-gate\n";
    if(control) { const auto e=need(a,"--engine");meta<<"engine_sha256="<<jrl::sha256(e)<<"\nengine_meta_sha256="<<jrl::sha256(e+".meta")<<'\n';std::filesystem::copy_file(e+".meta",directory+"/engine.meta"); }
    meta.close();if(!meta) throw std::runtime_error("metadata write failed");
    Serial serial(port);jrl_parser_t parser{};jrl::Observation obs;
    std::random_device random;const uint64_t session=(uint64_t(random())<<32)|random()|1u;
    uint64_t last_state=jrl::monotonic_us(),start=last_state;uint32_t reset=0,last_seq=0,pending=0;
    bool connected=false,selected_before=false;uint64_t seen_h7_session=0;
    uint64_t duration=a.count("--seconds") ? std::stoull(a.at("--seconds"))*1000000u:0;
    auto send=[&](uint8_t type,const uint8_t* source,const jrl::Action& action,uint32_t warmup,uint32_t us) {
        uint8_t p[68]={0};uint16_t n=type==JRL_ACTION?JRL_ACTION_BYTES:(type==JRL_HELLO?4:0);
        jrl_header(p,type,n,session,source?jrl_u32(source+16):reset,source?jrl_u32(source+20):last_seq,source?jrl_u64(source+24):0);
        if(type==JRL_HELLO) jrl_p32(p+32,JRL_CONTRACT);
        if(type==JRL_ACTION) { for(unsigned i=0;i<6;i++) jrl_pf32(p+32+4*i,action[i]);jrl_p32(p+56,warmup);jrl_p32(p+60,us); }
        unsigned bytes=jrl_finish(p);serial.send(p,bytes);log.push(2,p,bytes);
    };
    auto stop=[&] { if(control&&connected) { try { send(JRL_STOP,nullptr,{},0,0); } catch(...) {} } };
    std::signal(SIGINT,stop_signal);std::signal(SIGTERM,stop_signal);
    try {
        while(!stopping&&(!duration||jrl::monotonic_us()-start<duration)) {
            if(log.failed()) throw std::runtime_error("logger failed; stopping RL");
            if(control&&connected&&jrl::monotonic_us()-last_state>=100000u) throw std::runtime_error("H7 state timeout; rearm required");
            pollfd d{serial.fd,POLLIN,0};int rc=poll(&d,1,10);if(rc<0&&errno==EINTR) continue;if(rc<0||d.revents&(POLLERR|POLLHUP|POLLNVAL)) throw std::runtime_error("serial disconnected");
            uint8_t data[4096];ssize_t n=read(serial.fd,data,sizeof(data));if(n<0) { if(errno==EAGAIN||errno==EINTR) continue;throw std::runtime_error("serial read failed"); }if(!n) continue;
            log.push(1,data,size_t(n));
            std::vector<uint8_t> latest;unsigned state_count=0;uint32_t errors=parser.errors;
            for(ssize_t i=0;i<n;i++) { unsigned size=jrl_feed(&parser,data[i]);if(size&&parser.data[5]==JRL_STATE&&jrl_u16(parser.data+6)==JRL_STATE_BYTES) { latest.assign(parser.data,parser.data+size);state_count++; } }
            if(parser.errors!=errors) log.event("CRC/framing errors: "+std::to_string(parser.errors));
            if(latest.empty()) continue;
            if(state_count>1) log.event("backlog: skipped "+std::to_string(state_count-1)+" state frames");
            const uint8_t* h=latest.data();const uint8_t* p=h+32;uint32_t seq=jrl_u32(h+20),flags=jrl_u32(p),current_reset=jrl_u32(h+16);
            if(jrl_u32(p+24)!=JRL_CONTRACT) throw std::runtime_error("H7 contract mismatch");
            uint64_t h7session=jrl_u64(h+8);
            if(seq==last_seq&&h7session==seen_h7_session) continue;
            if(last_seq&&h7session==seen_h7_session&&uint32_t(seq-last_seq)!=1) log.event("state sequence gap: "+std::to_string(last_seq)+" -> "+std::to_string(seq));
            if(control&&connected&&h7session!=session) throw std::runtime_error("H7 session lost; rearm required");
            seen_h7_session=h7session;last_seq=seq;last_state=jrl::monotonic_us();
            if(!control) continue; // 只读录制不发送握手、动作或STOP。
            if(flags&JRL_BLOCKED) throw std::runtime_error("H7 blocked; disengage before reconnect");
            if(h7session!=session) { send(JRL_HELLO,h,{},0,0);continue; }
            connected=true;
            bool selected=(flags&(JRL_SELECTED|JRL_VALID))==(JRL_SELECTED|JRL_VALID);
            if(selected_before&&!selected&&reset==current_reset) throw std::runtime_error("active feedback became invalid; rearm required");
            if(reset!=current_reset||!selected) { reset=current_reset;obs.reset();pending=0;selected_before=false; }
            if(!selected) { send(JRL_HELLO,h,{},0,0);continue; }
            if(selected_before&&pending&&jrl_u32(p+8)!=pending) throw std::runtime_error("action not acknowledged; refusing history drift");
            if(obs.initialized()&&jrl_u32(p+12)<5u) { send(JRL_HELLO,h,{},0,0);continue; }
            bool warmup=!obs.initialized();auto input=obs.build(p);uint64_t before=jrl::monotonic_us();jrl::Action action{};
            if(!warmup) action=net->run(input);
            uint32_t us=uint32_t(jrl::monotonic_us()-before);for(float& f:action) f=jrl::clip(f,100.f);
            if(stopping) break;
            send(JRL_ACTION,h,action,warmup,us);pending=seq;selected_before=true;
            uint8_t record[640];jrl_p32(record,reset);jrl_p32(record+4,seq);jrl_p32(record+8,us);jrl_p32(record+12,warmup);
            for(unsigned i=0;i<150;i++) { jrl_pf32(record+16+4*i,input[i]); }
            for(unsigned i=0;i<6;i++) { jrl_pf32(record+616+4*i,action[i]); }
            log.push(3,record,sizeof(record));
        }
        stop();log.event("normal exit");
    } catch(const std::exception& e) { stop();log.event(std::string("fatal: ")+e.what());throw; }
}
}
int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::runtime_error("commands: convert, vectors, replay, record, control (explicit motor-control client)");
        Args a;for(int i=2;i<argc;i+=2) { if(i+1>=argc) throw std::runtime_error("options require values");a[argv[i]]=argv[i+1]; }
        std::string mode=argv[1];
        if(mode=="convert") jrl::Inference::convert(need(a,"--onnx"),need(a,"--engine"));
        else if(mode=="vectors"||mode=="replay") { jrl::Inference net(need(a,"--engine"));if(mode=="vectors") vectors(net,need(a,"--input"),need(a,"--output"));else replay(net,need(a,"--input"),need(a,"--output")); }
        else if(mode=="record"||mode=="control") live(a,mode=="control");
        else throw std::runtime_error("unknown command");
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
