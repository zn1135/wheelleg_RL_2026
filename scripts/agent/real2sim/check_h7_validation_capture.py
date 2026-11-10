"""Check the current C mirror packer and asynchronous USB sender for capture."""
import argparse
import json
from pathlib import Path
import tempfile

from scripts.agent.real2sim import check_real2sim_huananhu as check


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    check.H7=check.ROOT/'deployment/H7_RL_ros'
    with tempfile.TemporaryDirectory() as directory:
        directory=Path(directory)
        raw,packer=check.c_packets(directory,True)
        assert packer['protocol_version']==3
        result={'actual_h7_packer':packer,'stream':check.check_stream(raw),
                'usb_sender':check.check_usb_sender(directory,raw),
                'scope':'actual C packer and busy/commit USB stubs; no board or motion acceptance'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as file:json.dump(result,file,indent=2)
    print('Current v3 packer and asynchronous USB checks passed')


if __name__=='__main__':main()
