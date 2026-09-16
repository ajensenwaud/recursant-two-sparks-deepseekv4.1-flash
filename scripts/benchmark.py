#!/usr/bin/env python3
"""Single-stream sustained native-usage benchmark (not an answer-quality test).
Use the same prompt and settings for comparisons; no historical result is implied.
"""
import argparse,fcntl,hashlib,json,pathlib,re,statistics,time,urllib.parse,urllib.request
from verify import headers,stream,api_open


def usage_rate(samples):
    if len(samples)<2 or any(type(n) is not int or n<0 for t,n in samples):
        raise ValueError('two native completion-usage samples required')
    if any(b[0]<=a[0] or b[1]<a[1] for a,b in zip(samples,samples[1:])):
        raise ValueError('usage/timestamps must be monotonic')
    elapsed=samples[-1][0]-samples[0][0]
    return (samples[-1][1]-samples[0][1])/elapsed


def main():
    p=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p.add_argument('--prompt-file',required=True,type=pathlib.Path)
    p.add_argument('--output',required=True,type=pathlib.Path)
    p.add_argument('--seconds',type=int,default=60)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();base=a.base_url.rstrip('/');u=urllib.parse.urlsplit(base)
    if u.scheme not in ['http','https'] or u.username or u.password or u.query or u.fragment or u.path:
        raise ValueError('invalid endpoint')
    if u.scheme=='http' and u.hostname not in ['localhost','127.0.0.1','::1']:
        raise ValueError('use SSH forwarding or HTTPS; nonlocal plaintext benchmarking refused')
    if not 5<=a.seconds<=120 or not 1<=a.repeats<=5:raise ValueError('bounded 5..120 seconds, 1..5 repeats required')
    if a.prompt_file.stat().st_size>4*1024*1024:raise ValueError('prompt file cap')
    text=a.prompt_file.read_text();model='deepseek-ai/DeepSeek-V4.1-Flash'
    payload={'model':model,'messages':[{'role':'user','content':text}],
             'max_tokens':65536,'chat_template_kwargs':{'thinking':True,'reasoning_effort':'high'}}
    record={'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'seconds':a.seconds,'repeats':a.repeats,
            'concurrency':1,'output_allowance':65536,'thinking':'high','sampling':'server defaults',
            'warmup':'one excluded 32-token same-prefix request per repeat',
            'scope':'native usage window, deliberately cancelled; not completed-answer quality','windows':[]}
    if a.dry_run:print(json.dumps(record,indent=2));return
    if a.output.exists():raise ValueError('output already exists; preserve prior measurements')
    def idle():
        req=urllib.request.Request(base+'/metrics',headers=headers())
        with api_open(req,timeout=10) as r:metrics=r.read(8*1024*1024).decode()
        counts=[float(l.split()[-1]) for l in metrics.splitlines() if re.match(r'vllm:num_requests_(running|waiting)(\{|\s)',l)]
        return bool(counts) and all(n==0 for n in counts)
    lock_path=a.output.with_name(a.output.name+'.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not idle():raise ValueError('scheduler not idle')
        for repeat in range(a.repeats):
            stream(base,dict(payload,max_tokens=32),120) # excluded, may intentionally finish at length
            request=dict(payload,stream=True,stream_options={'include_usage':True,'continuous_usage_stats':True})
            req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(request).encode(),headers=headers())
            start=time.monotonic();samples=[];prompt_tokens=None;complete_window=False
            with api_open(req,timeout=30) as response:
                for raw in response:
                    now=time.monotonic()
                    if now-start>a.seconds+120:raise TimeoutError('benchmark request deadline')
                    if not raw.startswith(b'data:'):continue
                    data=raw[5:].strip()
                    if data==b'[DONE]':break
                    event=json.loads(data)
                    if event.get('error'):raise ValueError('streaming API error')
                    usage=event.get('usage') or {}
                    if type(usage.get('completion_tokens')) is int and usage['completion_tokens']>0:
                        samples.append((now,usage['completion_tokens']));prompt_tokens=usage.get('prompt_tokens')
                        if now-samples[0][0]>=a.seconds:complete_window=True;break
            window={'repeat':repeat+1,'complete_window':complete_window,'prompt_tokens':prompt_tokens,
                    'samples':samples,'native_usage_tps':usage_rate(samples)}
            record['windows'].append(window)
            a.output.write_text(json.dumps(record,indent=2)+'\n')
            for _ in range(30):
                if idle():break
                time.sleep(1)
            else:raise ValueError('scheduler failed to drain; stop benchmarking')
            if not complete_window:raise ValueError('early stop: incomplete sustained window retained')
        rates=[w['native_usage_tps'] for w in record['windows']]
        record.update(median_tps=statistics.median(rates),mean_tps=statistics.mean(rates))
        a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
