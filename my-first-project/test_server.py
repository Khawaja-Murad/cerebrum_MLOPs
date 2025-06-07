import argparse, os, requests

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--expected", type=int)
    args = p.parse_args()

    if not os.path.exists(args.image):
        print("Image missing:",args.image); exit(1)
    hdr = {"Authorization":f"Bearer {args.api_key}"}
    files = {"file":open(args.image,"rb")}
    r = requests.post(args.url,hdr,files=files)
    if r.status_code!=200:
        print("HTTP",r.status_code,r.text); exit(1)
    cid = r.json().get("class_id")
    print("→ class_id:",cid)
    if args.expected is not None and int(cid)!=args.expected:
        print(" MISMATCH exp",args.expected); exit(1)
    print("OK")

if __name__=="__main__":
    main()
