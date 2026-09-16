#!/usr/bin/env python3
"""Bounded actual text/reasoning/two-image API smoke; never executes outputs."""
import argparse,base64,json,os,re,struct,time,urllib.parse,urllib.request,zlib


def image_png(word,color,shape):
    glyphs={'C':['01110','10001','10000','10000','10000','10001','01110'],
            'A':['01110','10001','10001','11111','10001','10001','10001'],
            'T':['11111','00100','00100','00100','00100','00100','00100'],
            'D':['11110','10001','10001','10001','10001','10001','11110'],
            'O':['01110','10001','10001','10001','10001','10001','01110'],
            'G':['01110','10001','10000','10111','10001','10001','01110']}
    width,height=224,192;pixels=bytearray([255]*(width*height*3))
    def pixel(x,y,rgb):pixels[(y*width+x)*3:(y*width+x)*3+3]=bytes(rgb)
    for y in range(20,111):
        for x in range(67,158):
            if shape=='square' or (x-112)**2+(y-65)**2<=45**2:pixel(x,y,color)
    for i,letter in enumerate(word):
        for y,row in enumerate(glyphs[letter]):
            for x,bit in enumerate(row):
                if bit=='1':
                    for yy in range(6):
                        for xx in range(6):pixel(61+i*36+x*6+xx,132+y*6+yy,(0,0,0))
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    rows=b''.join(b'\0'+pixels[y*width*3:(y+1)*width*3] for y in range(height))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b'')


def parse_events(events,start):
    content=[];reasoning=False;first=None;last=None;usage=None;finish=None
    for now,event in events:
        if event.get('error'):raise ValueError('server streaming error')
        if event.get('usage'):usage=event['usage']
        for choice in event.get('choices',[]):
            delta=choice.get('delta',{});text=delta.get('content') or ''
            thought=delta.get('reasoning') or delta.get('reasoning_content') or ''
            if text or thought:
                first=now if first is None else first;last=now
            content.append(text);reasoning=reasoning or bool(thought)
            finish=choice.get('finish_reason') or finish
    if not usage or type(usage.get('completion_tokens')) is not int or usage['completion_tokens']<=0 or first is None:
        raise ValueError('missing authoritative native token usage or output')
    return {'content':''.join(content),'reasoning_observed':reasoning,'completion_tokens':usage['completion_tokens'],
            'finish_reason':finish,'ttft_seconds':first-start,
            'client_decode_tps_estimate':(usage['completion_tokens']-1)/(last-first) if last>first else None}


class NoAPIRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
        # Reject every hop before sending credentials or replaying a POST.
        raise ValueError('API redirects are disabled; configure the final endpoint')


def api_open(request, timeout):
    u = urllib.parse.urlsplit(request.full_url)
    if (u.scheme not in ['http','https'] or not u.hostname or u.username or
            u.password or u.fragment or u.query or not u.netloc or
            any(ord(c) <= 32 for c in request.full_url)):
        raise ValueError('invalid API URL')
    if u.port is not None and not 1 <= u.port <= 65535:
        raise ValueError('invalid API port')
    return urllib.request.build_opener(NoAPIRedirect()).open(request, timeout=timeout)


def headers():
    h={'Content-Type':'application/json'}
    if os.environ.get('RECIPE_API_KEY'):h['Authorization']='Bearer '+os.environ['RECIPE_API_KEY']
    return h


def stream(base,payload,timeout):
    payload=dict(payload,stream=True,stream_options={'include_usage':True,'continuous_usage_stats':True})
    start=time.monotonic();events=[];done=False;bytes_seen=0
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers=headers())
    with api_open(req,timeout=min(timeout,30)) as response:
        for raw in response:
            now=time.monotonic();bytes_seen+=len(raw)
            if now-start>timeout:raise TimeoutError('request wall deadline; incomplete, not a quality pass')
            if bytes_seen>16*1024*1024:raise ValueError('stream response cap')
            if not raw.startswith(b'data:'):continue
            data=raw[5:].strip()
            if data==b'[DONE]':done=True;break
            events.append((now,json.loads(data)))
    result=parse_events(events,start);result['wall_seconds']=time.monotonic()-start
    if not done:raise ValueError('stream ended without DONE')
    return result


def main():
    p=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p.add_argument('--allow-http-network',action='store_true')
    p.add_argument('--timeout',type=int,default=120)
    a=p.parse_args();base=a.base_url.rstrip('/');u=urllib.parse.urlsplit(base)
    if u.username or u.password or u.query or u.fragment or u.path or u.scheme not in ['http','https']:
        raise ValueError('invalid API base URL')
    if u.scheme=='http' and u.hostname not in ['127.0.0.1','localhost','::1'] and not a.allow_http_network:
        raise ValueError('non-loopback HTTP requires explicit --allow-http-network')
    if not 10<=a.timeout<=300:raise ValueError('timeout must be 10..300 seconds')
    def get(path):
        with api_open(urllib.request.Request(base+path,headers=headers()),timeout=15) as r:return r.read(1024*1024)
    get('/health');models=json.loads(get('/v1/models'))['data']
    expected='deepseek-ai/DeepSeek-V4.1-Flash'
    model=next((m for m in models if m['id']==expected),None)
    if model is None or model.get('max_model_len')!=1048576:raise ValueError('served model/context mismatch')
    cases=[('text','Reply with exactly OK.',['OK']),
           ('reasoning','Starting with 7, double it three times and then subtract 5. Reply with only the final integer.',['51'])]
    for word,color,shape in [('CAT',(255,0,0),'square'),('DOG',(0,0,255),'circle')]:
        data='data:image/png;base64,'+base64.b64encode(image_png(word,color,shape)).decode()
        content=[{'type':'text','text':'Name the color, geometric shape, and printed word in this image. Be brief.'},
                 {'type':'image_url','image_url':{'url':data}}]
        cases.append(('vision-'+word.lower(),content,['red' if word=='CAT' else 'blue',shape,word.lower()]))
    results=[]
    for name,content,expected_words in cases:
        r=stream(base,{'model':model['id'],'messages':[{'role':'user','content':content}],
                       'temperature':0,'max_tokens':65536,
                       'chat_template_kwargs':{'thinking':True,'reasoning_effort':'high'}},a.timeout)
        answer=r.pop('content').strip()
        correct=(answer==expected_words[0] if name in ['text','reasoning'] else all(w in answer.lower() for w in expected_words))
        results.append({'name':name,'correct':correct,**r})
        if not correct or r['finish_reason']!='stop':
            print(json.dumps({'passed':False,'results':results},indent=2));raise SystemExit(1)
    print(json.dumps({'passed':True,'scope':'four bounded endpoint smokes, not general model certification',
                      'results':results},indent=2))
if __name__=='__main__':main()
