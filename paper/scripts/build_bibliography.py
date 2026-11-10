import json,unicodedata
from pathlib import Path
r=Path(__file__).resolve().parents[1];d=json.loads((r/'evidence/reference-metadata.json').read_text())
def ascii(s):return unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().replace('&',r'\&')
entries=[]
for key,v in d.items():
 assert 'title' in v,key
 typ='inproceedings' if key in ['gru','gesture'] else 'article'
 fields={'title':'{'+ascii(v['title'][0])+'}','author':' and '.join(ascii(a['family'])+', '+ascii(a.get('given','')) for a in v['author']),'year':str(v['published']['date-parts'][0][0]),'booktitle' if typ=='inproceedings' else 'journal':ascii(v['container-title'][0]),'doi':v['DOI']}
 for k in ['volume','issue','page']:
  if v.get(k):fields[{'issue':'number','page':'pages'}.get(k,k)]=str(v[k]).replace('-', '--') if k=='page' else str(v[k])
 entries.append('@'+typ+'{'+key+',\n'+',\n'.join('  '+k+' = {'+val+'}' for k,val in fields.items())+'\n}')
entries.append(r'''@misc{dmp,
 author={Sun, Pengfei and Su, Zhe and Achterberg, Jascha and Indiveri, Giacomo and Goodman, Dan F. M. and Akarca, Danyal},
 title={{Algorithm-hardware co-design of neuromorphic networks with dual memory pathways}},
 year={2026}, howpublished={arXiv:2512.07602v3}, url={https://arxiv.org/abs/2512.07602v3}
}
@inproceedings{lmu,
 author={Voelker, Aaron and Kajic, Ivana and Eliasmith, Chris},
 title={{Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks}},
 booktitle={Advances in Neural Information Processing Systems},volume={32},year={2019},
 url={https://papers.nips.cc/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html}
}
@inproceedings{hippo,
 author={Gu, Albert and Dao, Tri and Ermon, Stefano and Rudra, Atri and Re, Christopher},
 title={{HiPPO: Recurrent Memory with Optimal Polynomial Projections}},
 booktitle={Advances in Neural Information Processing Systems},volume={33},pages={1474--1487},year={2020}
}
@inproceedings{s4,
 author={Gu, Albert and Goel, Karan and Re, Christopher},
 title={{Efficiently Modeling Long Sequences with Structured State Spaces}},
 booktitle={International Conference on Learning Representations},year={2022},url={https://arxiv.org/abs/2111.00396}
}
@misc{mamba,
 author={Gu, Albert and Dao, Tri},
 title={{Mamba: Linear-Time Sequence Modeling with Selective State Spaces}},
 year={2023},howpublished={arXiv:2312.00752},url={https://arxiv.org/abs/2312.00752}
}
@inproceedings{glif,
 author={Yao, X. and Li, F. and Mo, Z. and Cheng, J.},
 title={{GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks}},
 booktitle={Advances in Neural Information Processing Systems},volume={35},pages={32160--32171},year={2022}
}
@inproceedings{tclif,
 author={Zhang, S. and Yang, Q. and Ma, C. and Wu, J. and Li, H. and Tan, K. C.},
 title={{TC-LIF: A Two-Compartment Spiking Neuron Model for Long-term Sequential Modelling}},
 booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},volume={38},pages={16838--16847},year={2024}
}
''')
(r/'references.bib').write_text('\n\n'.join(entries))
print('Wrote',len(d)+7,'references')
