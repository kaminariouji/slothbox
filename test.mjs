
import { pipeline } from '@huggingface/transformers';
const t0 = Date.now();
const embed = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8' });
console.log('load ms', Date.now()-t0);
const out = await embed(['a sloth moves slowly through the canopy', 'bitcoin mining hardware'], { pooling: 'mean', normalize: true });
console.log('dim', out.dims, 'first3', Array.from(out.data).slice(0,3).map(x=>x.toFixed(4)));
const a = Array.from(out.data).slice(0,384), b = Array.from(out.data).slice(384,768);
const dot = a.reduce((s,v,i)=>s+v*b[i],0);
console.log('cos', dot.toFixed(4));

