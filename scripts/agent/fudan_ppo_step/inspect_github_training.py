"""Pin and read selected public training sources; never imports or executes downloaded code."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def read(url):
    request=urllib.request.Request(url,headers={'User-Agent':'Wheel-Legged-Gym-source-review'})
    with urllib.request.urlopen(request,timeout=45) as response:return response.read()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='mode',required=True)
    t=sub.add_parser('tree');t.add_argument('--repo',required=True);t.add_argument('--revision')
    t.add_argument('--output',type=Path,required=True)
    f=sub.add_parser('fetch');f.add_argument('--snapshot',type=Path,required=True);f.add_argument('--paths',nargs='+',required=True)
    args=p.parse_args()
    if args.mode=='tree':
        args.output.mkdir(parents=True,exist_ok=False)
        revision=args.revision or json.loads(read(f'https://api.github.com/repos/{args.repo}/commits?per_page=1'))[0]['sha']
        tree=json.loads(read(f'https://api.github.com/repos/{args.repo}/git/trees/{revision}?recursive=1'))
        assert not tree.get('truncated')
        write(args.output/'tree.json',tree)
        write(args.output/'manifest.json',dict(repository=args.repo,commit=revision,files={},executed=False))
        selected=[entry['path'] for entry in tree['tree'] if entry['type']=='blob' and
                  any(word in entry['path'].lower() for word in ('reward','curricul','terrain','config','readme')) and
                  entry['path'].endswith(('.py','.md','.yaml','.yml'))]
        print(json.dumps(dict(repo=args.repo,commit=revision,paths=selected),ensure_ascii=False))
    else:
        manifest=json.loads((args.snapshot/'manifest.json').read_text())
        index={entry['path']:entry for entry in json.loads((args.snapshot/'tree.json').read_text())['tree'] if entry['type']=='blob'}
        for name in args.paths:
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Relative repository path required')
            url=f'https://raw.githubusercontent.com/{manifest["repository"]}/{manifest["commit"]}/'+urllib.parse.quote(name)
            data=read(url)
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==index[name]['sha']
            target=args.snapshot/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(data)
            manifest['files'][name]=dict(url=url,sha256=hashlib.sha256(data).hexdigest(),git_blob=index[name]['sha'])
        write(args.snapshot/'manifest.json',manifest)
        print(json.dumps(dict(repository=manifest['repository'],commit=manifest['commit'],files=len(manifest['files']),executed=False)))


if __name__=='__main__':
    main()
