import urllib.request,json,concurrent.futures,pathlib
root=pathlib.Path(__file__).resolve().parents[1]
dois={'lstm':'10.1162/neco.1997.9.8.1735','shd':'10.1109/TNNLS.2020.3044364','mnist':'10.1109/5.726791','gesture':'10.1109/CVPR.2017.781','surrogate':'10.1109/MSP.2019.2931595','spikingjelly':'10.1126/sciadv.adi1480','gru':'10.3115/v1/D14-1179'}
def fetch(kv):
 k,d=kv
 try:
  req=urllib.request.Request('https://api.crossref.org/works/'+d,headers={'User-Agent':'ManuscriptMetadataCheck/1.0'})
  with urllib.request.urlopen(req,timeout=25) as r:v=json.load(r)['message']
  return k,{key:v.get(key) for key in ['DOI','title','author','container-title','published','published-print','published-online','volume','issue','page','URL']}
 except Exception as e:return k,{'error':str(e),'DOI':d}
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:result=dict(ex.map(fetch,dois.items()))
(root/'evidence/reference-metadata.json').write_text(json.dumps(result,indent=2))
for k,v in result.items():print(k,v.get('title'),v.get('published'),v.get('error',''))
