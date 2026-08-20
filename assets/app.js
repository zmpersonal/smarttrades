const $=(s,p=document)=>p.querySelector(s); const $$=(s,p=document)=>[...p.querySelectorAll(s)];
function pctDecimal(n,d=1){return n==null?'—':(+n*100).toFixed(d)+'%'}
async function publicData(){try{return await fetch('/data/public.json?'+Date.now()).then(r=>r.json())}catch(e){return null}}
function renderRows(el,rows){if(!el)return;el.innerHTML=(rows||[]).map((x,i)=>`<tr><td>${i+1}</td><td><a class="ticker" href="/stock/?ticker=${encodeURIComponent(x.ticker||'')}">${x.ticker||'—'}</a><div class="tiny">${x.name||''}</div></td><td><span class="score">${Math.round(x.score||0)}</span></td><td>${Math.round(x.smart_score||0)}</td><td>${x.sector||'—'}</td></tr>`).join('')||'<tr><td colspan="5" class="muted">Ranking data has not been initialized yet.</td></tr>'}
async function initRanks(){let d=await publicData(); $$('[data-ranking]').forEach(el=>{let slug=el.dataset.ranking;renderRows(el.querySelector('tbody'),d?.rankings?.[slug]?.slice(0,10));let stamp=el.querySelector('[data-stamp]');if(stamp)stamp.textContent=d?.updated_at?`Updated ${new Date(d.updated_at).toLocaleString()} · ${d.coverage_count||0} companies covered`:'Awaiting first SEC refresh'});let demo=$('.demo');if(demo&&d?.is_demo===false)demo.remove();let coverage=$('[data-coverage]');if(coverage&&d?.coverage_label)coverage.textContent=d.coverage_label}
async function screener(){let root=$('#screener-results');if(!root)return;let d=await publicData();let rows=d?.public_universe||[];function run(){let q=($('#q')?.value||'').toLowerCase(),min=+($('#minscore')?.value||0),sec=$('#sector')?.value||'',style=$('#style')?.value||'';let f=rows.filter(x=>(!q||(x.ticker+' '+x.name).toLowerCase().includes(q))&&(+x.smart_score>=min)&&(!sec||x.sector===sec)&&(!style||x.tags?.includes(style)));root.innerHTML=f.map(x=>`<tr><td><a class="ticker" href="/stock/?ticker=${encodeURIComponent(x.ticker)}">${x.ticker}</a><div class="tiny">${x.name}</div></td><td><span class="score">${Math.round(x.smart_score)}</span></td><td>${Math.round(x.quality_score)}</td><td>${Math.round(x.growth_score)}</td><td>${Math.round(x.financial_strength_score)}</td></tr>`).join('')||'<tr><td colspan="5">No matches.</td></tr>'}$$('[data-filter]').forEach(x=>x.addEventListener('input',run));run()}
async function stockDetail(){let root=$('#stock-detail');if(!root)return;let t=(new URLSearchParams(location.search).get('ticker')||'').toUpperCase();let d=await publicData();let x=(d?.public_universe||[]).find(z=>z.ticker===t);if(!x){root.innerHTML='<div class="lock"><h3>Stock not in the current free-beta universe</h3><p class="muted">SmartTrades currently covers a deliberately small set of large U.S. operating companies using standardized SEC filing data.</p><a class="btn primary" href="/screener/">Open screener</a></div>';return}root.innerHTML=`<div class="grid grid2"><div class="card"><div class="eyebrow">Smart Score</div><div class="kpi">${Math.round(x.smart_score||0)}/100</div><h2>${x.ticker} · ${x.name||''}</h2><p class="muted">${x.sector||''}</p><div class="tiny">Latest fiscal period: ${x.latest_fiscal_end||'—'}</div></div><div class="card"><h2>Factor snapshot</h2><div class="metric"><span>Quality</span><strong>${Math.round(x.quality_score||0)}</strong></div><div class="metric"><span>Growth</span><strong>${Math.round(x.growth_score||0)}</strong></div><div class="metric"><span>Financial strength</span><strong>${Math.round(x.financial_strength_score||0)}</strong></div><div class="metric"><span>3Y revenue growth</span><strong>${pctDecimal(x.revenue_growth_3y)}</strong></div><div class="metric"><span>3Y FCF growth</span><strong>${pctDecimal(x.fcf_growth_3y)}</strong></div><div class="metric"><span>FCF margin</span><strong>${pctDecimal(x.fcf_margin)}</strong></div><div class="metric"><span>3Y dividend growth</span><strong>${pctDecimal(x.dividend_growth_3y)}</strong></div></div></div><div class="notice" style="margin-top:18px">Free beta uses SEC filing fundamentals only. Live price, valuation, analyst estimates and momentum are intentionally excluded until SmartTrades has a suitable licensed market-data source.</div>`}
screener();initRanks();stockDetail();
\n\nfunction initLeadForms(){
  const api=(window.SMARTTRADES_CONFIG&&window.SMARTTRADES_CONFIG.apiBase||'').replace(/\/$/,'');
  $$('[data-lead-form]').forEach(form=>{
    const status=$('[data-lead-status]',form);
    form.addEventListener('submit',async e=>{
      e.preventDefault();
      const email=String(new FormData(form).get('email')||'').trim();
      const source=form.dataset.source||location.pathname||'site';
      if(!email||!email.includes('@')){if(status)status.textContent='Enter a valid email address.';return;}
      const btn=$('button[type="submit"]',form); if(btn){btn.disabled=true;btn.textContent='Joining…'}
      if(status)status.textContent='';
      try{
        const r=await fetch(api+'/api/leads',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email,source})});
        const j=await r.json().catch(()=>({}));
        if(!r.ok)throw new Error(j.error||'Could not save your email.');
        form.reset();
        if(status)status.textContent='You’re on the SmartTrades list.';
      }catch(err){if(status)status.textContent=err.message||'Something went wrong. Please try again.'}
      finally{if(btn){btn.disabled=false;btn.textContent='Join free'}}
    });
  });
}
initLeadForms();
