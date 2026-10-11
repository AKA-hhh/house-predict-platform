/* ==========================================================================
   二手房价格分析与预测系统 — 前端应用
   单页应用：11 个功能页，对应三大模块（数据处理 / 核心算法 / 结果展示）
   ========================================================================== */
'use strict';

const API = '';
const S = {
  tab: 'overview',
  health: null,
  sources: [],
  versions: [],
  activeVersion: null,
  agg: null,
  algorithms: [],
  tasks: [],
  models: [],
  docs: [],
  chartTab: null,
};

/* ------------------------------------------------------------ 基础工具 */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function h(html) { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; }
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function num(v, d = 0) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  return Number(v).toLocaleString('zh-CN', { minimumFractionDigits: d, maximumFractionDigits: d });
}
function wan(v) { return v === null || v === undefined ? '—' : num(v, 1) + ' 万'; }
function pct(v, d = 2) { return v === null || v === undefined ? '—' : Number(v).toFixed(d) + '%'; }
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

function toast(msg, type = '', ms = 3600) {
  const t = h(`<div class="toast ${type}"><span>${type === 'ok' ? '✓' : type === 'err' ? '✕' : type === 'warn' ? '!' : 'ℹ'}</span><span>${esc(msg)}</span></div>`);
  $('#toasts').appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .3s'; setTimeout(() => t.remove(), 320); }, ms);
}

function modal(title, bodyHtml, wide = false) {
  const m = h(`<div class="modal-mask"><div class="modal" ${wide ? 'style="max-width:1120px"' : ''}>
      <div class="modal-head"><div class="card-title">${esc(title)}</div>
      <button class="btn btn-ghost btn-sm" data-close>关闭 ✕</button></div>
      <div data-body></div></div></div>`);
  $('[data-body]', m).innerHTML = bodyHtml;
  m.addEventListener('click', e => { if (e.target === m || e.target.hasAttribute('data-close')) m.remove(); });
  document.body.appendChild(m);
  return m;
}

async function api(method, path, body) {
  const opt = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) opt.body = JSON.stringify(body);
  const r = await fetch(API + path, opt);
  const text = await r.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch (e) { data = { detail: text }; }
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
  return data;
}
const GET = (p) => api('GET', p);
const POST = (p, b) => api('POST', p, b);

/* ------------------------------------------------------------ 图表管理 */
const charts = {};
function ec(id, option, height) {
  const dom = document.getElementById(id);
  if (!dom) return null;
  if (height) dom.style.height = height;
  if (charts[id]) { charts[id].dispose(); }
  const c = echarts.init(dom, null, { renderer: 'canvas' });
  c.setOption(option);
  charts[id] = c;
  return c;
}
window.addEventListener('resize', () => Object.values(charts).forEach(c => c.resize()));

const PALETTE = ['#1d5fd0', '#0e9aa7', '#c98a2e', '#7a5af8', '#d93a34', '#178f57', '#e0679b', '#5b7c99'];
const AXIS = {
  axisLine: { lineStyle: { color: '#e2e7f0' } },
  axisLabel: { color: '#8a95a8', fontSize: 11 },
  splitLine: { lineStyle: { color: '#eef1f7' } },
};
const TIP = { trigger: 'axis', backgroundColor: '#16202f', borderWidth: 0, textStyle: { color: '#fff', fontSize: 12 }, axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(29,95,208,.05)' } } };
const GRID = { left: 54, right: 22, top: 34, bottom: 34 };

/* ============================================================ 页面元信息 */
const PAGES = {
  overview: ['数据概览', '数据集规模、价格分布、区域与时间统计'],
  pipeline: ['数据导入与流水线', '数据源配置 · 导入 · 清洗 · 特征 · 划分 · 存储'],
  versions: ['数据版本记录', '每次流水线运行的参数、行数、校验和与产物'],
  quality: ['清洗与质量报告', '缺失值 / 异常值 / 重复值处理明细与数据画像'],
  listings: ['房源明细查询', '按区域、价格、面积、户型条件检索房源'],
  algorithms: ['算法与参数配置', '系统内全部算法的输入 / 输出 / 参数 / 评价方法'],
  training: ['模型训练与对比', '选择算法与参数，训练并比较价格预测效果'],
  predict: ['价格预测', '输入房源属性，输出预测单价与可解释性归因'],
  recommend: ['房源推荐', '内容推荐 / 协同过滤 / 混合推荐 / 相似房源检索'],
  experiments: ['实验与运行记录', '真实运行结果、对比数据与失败案例'],
  docs: ['项目文档', '数据来源、系统设计、AI 辅助编程记录、实验分析'],
};

/* ============================================================ 启动 */
async function boot() {
  $$('.nav-item').forEach(n => n.addEventListener('click', () => go(n.dataset.tab)));
  window.addEventListener('hashchange', () => {
    const t = (location.hash || '').replace('#', '');
    if (t && PAGES[t] && t !== S.tab) go(t);
  });
  const initial = (location.hash || '').replace('#', '');
  go(PAGES[initial] ? initial : 'overview');
  refreshHealth();
  setInterval(refreshHealth, 30000);
}

async function refreshHealth() {
  try {
    const hl = await GET('/api/health');
    S.health = hl;
    $('#health-dot').className = 'dot ok';
    $('#health-text').textContent = `正常 · 数据 ${num(hl.detail_rows)} 行 · 模型 ${hl.models} 个`;
  } catch (e) {
    $('#health-dot').className = 'dot';
    $('#health-text').textContent = '服务未就绪';
  }
}

function go(tab) {
  S.tab = tab;
  S.chartTab = tab;
  if (location.hash !== '#' + tab) history.replaceState(null, '', '#' + tab);
  $$('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.tab === tab));
  const [t, d] = PAGES[tab];
  $('#ptitle').textContent = t;
  $('#pdesc').textContent = d;
  $('#topbar-actions').innerHTML = '';
  Object.keys(charts).forEach(k => { charts[k].dispose(); delete charts[k]; });
  const c = $('#content');
  c.innerHTML = `<div class="card"><div class="skeleton" style="height:120px"></div></div>`;
  (RENDER[tab] || (() => { }))().catch(err => {
    c.innerHTML = `<div class="card"><div class="alert alert-err"><span>✕</span><div><b>页面加载失败</b><br>${esc(err.message)}</div></div></div>`;
  });
}

/* ============================================================ 1. 数据概览 */
async function renderOverview() {
  const c = $('#content');
  // 概览页需要用到直方图/散点/相关性/地理点等全部预聚合结果，
  // 因此直接取完整汇总（一次请求拿全，避免多次往返）
  const agg = await GET('/api/analytics/aggregates').catch(() => null);
  if (!agg || !agg.basic) {
    c.innerHTML = emptyState('尚未运行数据流水线', '系统还没有可展示的数据。请先进入「数据导入与流水线」，选择数据源并点击「运行流水线」。',
      `<button class="btn btn-primary" onclick="go('pipeline')">前往数据导入 →</button>`);
    return;
  }
  const ov = agg;
  const b = ov.basic;
  S.agg = agg;

  c.innerHTML = `
    <div class="grid g4" style="margin-bottom:18px">
      ${kpi('房源记录数', num(b.rows), '条', '清洗后可用样本', '')}
      ${kpi('平均单价', num(b.avg_unit_price), '元/㎡', `中位数 ${num(b.median_unit_price)}`, 'k-red')}
      ${kpi('平均总价', num(b.avg_total_price, 1), '万', `平均面积 ${num(b.avg_area, 1)} ㎡`, 'k-gold')}
      ${kpi('覆盖区域 / 小区', `${b.n_districts} / ${num(b.n_communities)}`, '', `成交区间 ${(b.trade_period || ['', ''])[0]} ~ ${(b.trade_period || ['', ''])[1]}`, 'k-purple')}
    </div>
    <div class="grid g4" style="margin-bottom:18px">
      ${kpi('单价 P10–P90', `${num(b.p10_unit_price)} – ${num(b.p90_unit_price)}`, '元/㎡', '价格离散区间', 'k-green')}
      ${kpi('平均楼龄', num(b.avg_building_age, 1), '年', '成交时点计算', '')}
      ${kpi('平均挂牌天数', num(b.avg_days_on_market, 1), '天', 'Days on Market', 'k-gold')}
      ${kpi('平均关注人数', num(b.avg_followers, 1), '人', '热度代理指标', 'k-purple')}
    </div>

    <div class="grid g2-21">
      <div class="card"><div class="card-head"><div><div class="card-title">各区平均单价</div>
        <div class="card-sub">按 48,000 元/㎡ 参考线着色，红=高于参考，绿=低于参考</div></div></div>
        <div id="ch-district" class="chart chart-lg"></div></div>
      <div class="card"><div class="card-head"><div class="card-title">房源结构</div></div>
        <div id="ch-struct" class="chart chart-lg"></div></div>
    </div>

    <div class="grid g2">
      <div class="card"><div class="card-head"><div><div class="card-title">单价分布</div>
        <div class="card-sub">直方图（40 组距），展示价格集中度与长尾</div></div></div>
        <div id="ch-hist" class="chart"></div></div>
      <div class="card"><div class="card-head"><div><div class="card-title">面积 — 单价散点</div>
        <div class="card-sub">抽样 1,200 条，观察面积与单价的关系及离群点</div></div></div>
        <div id="ch-scatter" class="chart"></div></div>
    </div>

    <div class="card"><div class="card-head"><div class="card-title">成交均价趋势</div>
      <div class="card-sub">按成交月份聚合，反映北京二手房市场价格走势（2011-11 ~ 2017-02）</div></div>
      <div id="ch-trend" class="chart chart-lg"></div></div>

    <div class="grid g2-13">
      <div class="card"><div class="card-head"><div class="card-title">地理分布</div>
        <div class="card-sub">抽样 2,000 条经纬度，颜色深浅表示单价高低</div></div>
        <div id="ch-geo" class="chart chart-lg"></div></div>
      <div class="card"><div class="card-head"><div class="card-title">特征相关性</div>
        <div class="card-sub">皮尔逊相关系数矩阵（红正绿负）</div></div>
        <div id="ch-corr" class="chart chart-lg"></div></div>
    </div>

    <div class="grid g2">
      <div class="card"><div class="card-head"><div class="card-title">面积区间与单价</div></div>
        <div id="ch-area" class="chart"></div></div>
      <div class="card"><div class="card-head"><div class="card-title">装修 / 电梯 / 地铁 分组</div></div>
        <div id="ch-flags" class="chart"></div></div>
    </div>
  `;

  // 各区平均单价
  const d = (ov.district || []).slice(0, 13);
  ec('ch-district', {
    grid: { ...GRID, bottom: 78, left: 62 },
    tooltip: { ...TIP, formatter: p => `${p[0].name}<br/>平均单价 <b>${num(p[0].value)}</b> 元/㎡<br/>样本 ${num((d[p[0].dataIndex] || {}).count)} 条` },
    xAxis: { type: 'category', data: d.map(x => x.district_name), ...AXIS, axisLabel: { ...AXIS.axisLabel, rotate: 38 } },
    yAxis: { type: 'value', ...AXIS, axisLabel: { ...AXIS.axisLabel, formatter: v => (v / 10000).toFixed(0) + '万' } },
    series: [{
      type: 'bar', data: d.map(x => ({ value: x.avg_unit_price, itemStyle: { color: x.avg_unit_price >= 48000 ? '#d93a34' : '#178f57' } })),
      barMaxWidth: 30, itemStyle: { borderRadius: [4, 4, 0, 0] },
      label: { show: true, position: 'top', fontSize: 10, color: '#55637a', formatter: p => num(p.value / 1000, 0) + 'k' },
    }],
  });

  // 房源结构（区域占比）
  ec('ch-struct', {
    tooltip: { ...TIP, trigger: 'item', formatter: p => `${p.name}<br/>${num(p.value)} 条 (${p.percent}%)` },
    legend: { type: 'scroll', orient: 'vertical', right: 6, top: 'middle', textStyle: { fontSize: 11, color: '#55637a' } },
    series: [{
      type: 'pie', radius: ['40%', '68%'], center: ['36%', '52%'], avoidLabelOverlap: true,
      itemStyle: { borderColor: '#fff', borderWidth: 2 },
      label: { show: false }, labelLine: { show: false },
      data: d.map((x, i) => ({ name: x.district_name, value: x.count, itemStyle: { color: PALETTE[i % PALETTE.length] } })),
    }],
  });

  // 单价分布
  const hh = ov.hist_unit_price || [];
  ec('ch-hist', {
    grid: { ...GRID, top: 22, bottom: 38 },
    tooltip: { ...TIP, formatter: p => `${num(p[0].value.bin_start)} – ${num(p[0].value.bin_end)} 元/㎡<br/>${num(p[0].value.count)} 条` },
    xAxis: { type: 'category', data: hh.map(x => num(x.bin_start / 1000, 0)), ...AXIS, name: '千元/㎡', nameTextStyle: { fontSize: 10, color: '#8a95a8' } },
    yAxis: { type: 'value', ...AXIS },
    series: [{ type: 'bar', data: hh.map(x => x.count), itemStyle: { color: '#1d5fd0', borderRadius: [3, 3, 0, 0] }, barCategoryGap: '12%' }],
  });

  // 散点
  const sc = (S.agg.scatter_area_price) || [];
  ec('ch-scatter', {
    grid: { ...GRID, top: 22, right: 30 },
    tooltip: { ...TIP, trigger: 'item', formatter: p => `${p.data[2]}<br/>面积 ${p.data[0]} ㎡<br/>单价 ${num(p.data[1])} 元/㎡<br/>总价 ${p.data[3]} 万` },
    xAxis: { type: 'value', name: '面积 ㎡', ...AXIS, nameTextStyle: { fontSize: 10, color: '#8a95a8' } },
    yAxis: { type: 'value', name: '单价 元/㎡', ...AXIS, nameTextStyle: { fontSize: 10, color: '#8a95a8' } },
    series: [{ type: 'scatter', symbolSize: 4, data: sc.map(x => [x.area, x.unit_price, x.district, x.total_price]), itemStyle: { color: 'rgba(29,95,208,.4)' } }],
  });

  // 趋势
  const tr = ov.trend_monthly || [];
  ec('ch-trend', {
    grid: { ...GRID, top: 38, bottom: 40 },
    tooltip: { ...TIP, formatter: p => `${p[0].name}<br/>平均单价 <b>${num(p[0].value)}</b> 元/㎡<br/>成交 ${num((tr[p[0].dataIndex] || {}).count)} 套` },
    legend: { top: 2, textStyle: { fontSize: 11, color: '#55637a' } },
    xAxis: { type: 'category', data: tr.map(x => x.period), ...AXIS },
    yAxis: [
      { type: 'value', name: '元/㎡', ...AXIS, nameTextStyle: { fontSize: 10, color: '#8a95a8' } },
      { type: 'value', name: '成交量', ...AXIS, nameTextStyle: { fontSize: 10, color: '#8a95a8' }, splitLine: { show: false } },
    ],
    series: [
      { name: '平均单价', type: 'line', smooth: true, symbol: 'none', data: tr.map(x => x.avg_unit_price), lineStyle: { width: 2.6, color: '#d93a34' }, itemStyle: { color: '#d93a34' }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(217,58,52,.16)' }, { offset: 1, color: 'rgba(217,58,52,0)' }]) } },
      { name: '成交量', type: 'bar', yAxisIndex: 1, data: tr.map(x => x.count), itemStyle: { color: 'rgba(29,95,208,.28)', borderRadius: [3, 3, 0, 0] }, barMaxWidth: 14 },
    ],
  });

  // 地理
  const gp = (S.agg.geo_points) || [];
  const prices = gp.map(x => x.unit_price);
  const pmax = Math.max(...prices, 1);
  ec('ch-geo', {
    grid: { ...GRID, top: 20, right: 26, bottom: 44 },
    tooltip: { ...TIP, trigger: 'item', formatter: p => `单价 ${num(p.data[2])} 元/㎡` },
    xAxis: { type: 'value', name: '经度', ...AXIS, min: 115.9, max: 117.0, nameTextStyle: { fontSize: 10, color: '#8a95a8' } },
    yAxis: { type: 'value', name: '纬度', ...AXIS, min: 39.6, max: 40.4, nameTextStyle: { fontSize: 10, color: '#8a95a8' } },
    visualMap: { show: true, dimension: 2, min: 10000, max: 90000, right: 8, top: 8, calculable: true, itemWidth: 11, itemHeight: 92, textStyle: { fontSize: 10, color: '#8a95a8' }, inRange: { color: ['#178f57', '#c9d84a', '#e8a33d', '#d93a34'] } },
    series: [{ type: 'scatter', symbolSize: 3.4, data: gp.map(x => [x.lng, x.lat, x.unit_price]), itemStyle: { opacity: .58 } }],
  });

  // 相关性
  const co = ov.correlation || { columns: [], matrix: [] };
  const cd = [];
  co.matrix.forEach((row, i) => row.forEach((v, j) => cd.push([j, i, v])));
  ec('ch-corr', {
    grid: { left: 132, right: 24, top: 16, bottom: 92 },
    tooltip: { ...TIP, trigger: 'item', formatter: p => `${co.columns[p.data[1]]} × ${co.columns[p.data[0]]}<br/>r = <b>${p.data[2]}</b>` },
    xAxis: { type: 'category', data: co.columns, ...AXIS, axisLabel: { ...AXIS.axisLabel, rotate: 52 } },
    yAxis: { type: 'category', data: co.columns, ...AXIS },
    visualMap: { min: -1, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: 4, itemWidth: 11, itemHeight: 92, textStyle: { fontSize: 10, color: '#8a95a8' }, inRange: { color: ['#178f57', '#f3f6fa', '#d93a34'] } },
    series: [{ type: 'heatmap', data: cd, label: { show: co.columns.length <= 14, fontSize: 9, color: '#16202f', formatter: p => p.data[2].toFixed(2) }, itemStyle: { borderColor: '#fff', borderWidth: 1 } }],
  });

  // 面积区间
  const ab = ov.area_bucket || [];
  ec('ch-area', {
    grid: { ...GRID, bottom: 32 },
    tooltip: { ...TIP, formatter: p => `${p[0].name} ㎡<br/>平均单价 <b>${num(p[0].value)}</b> 元/㎡<br/>样本 ${num((ab[p[0].dataIndex] || {}).count)}` },
    xAxis: { type: 'category', data: ab.map(x => x.area_bucket), ...AXIS },
    yAxis: { type: 'value', ...AXIS },
    series: [{ type: 'bar', data: ab.map(x => x.avg_unit_price), itemStyle: { color: '#0e9aa7', borderRadius: [4, 4, 0, 0] }, barMaxWidth: 42, label: { show: true, position: 'top', fontSize: 10, color: '#55637a', formatter: p => num(p.value / 1000, 0) + 'k' } }],
  });

  // 装修/电梯/地铁
  const ren = ov.renovation || [], el2 = ov.elevator || [], sw = ov.subway || [];
  ec('ch-flags', {
    grid: { ...GRID, bottom: 30 },
    tooltip: { ...TIP, formatter: p => `${p[0].name}<br/>平均单价 <b>${num(p[0].value)}</b> 元/㎡` },
    xAxis: { type: 'category', data: [...ren.map(x => x.renovation_name), ...el2.map(x => x.elevator_name), ...sw.map(x => x.subway_name)], ...AXIS, axisLabel: { ...AXIS.axisLabel, rotate: 22 } },
    yAxis: { type: 'value', ...AXIS },
    series: [{
      type: 'bar', barMaxWidth: 30, itemStyle: { borderRadius: [4, 4, 0, 0] },
      data: [...ren.map(x => ({ value: x.avg_unit_price, itemStyle: { color: '#c98a2e' } })),
      ...el2.map(x => ({ value: x.avg_unit_price, itemStyle: { color: '#1d5fd0' } })),
      ...sw.map(x => ({ value: x.avg_unit_price, itemStyle: { color: '#7a5af8' } }))],
    }],
  });
}

function kpi(label, value, unit, hint, cls) {
  return `<div class="kpi ${cls || ''}"><div class="kpi-label">${esc(label)}</div>
    <div class="kpi-value">${value}${unit ? `<small>${esc(unit)}</small>` : ''}</div>
    <div class="kpi-hint">${esc(hint || '')}</div></div>`;
}
function emptyState(title, desc, action = '') {
  return `<div class="card"><div class="empty"><div class="big">◍</div>
    <h3>${esc(title)}</h3><p>${esc(desc)}</p>${action}</div></div>`;
}

/* ============================================================ 2. 数据导入与流水线 */
async function renderPipeline() {
  const c = $('#content');
  const [srcs, def] = await Promise.all([GET('/api/sources'), GET('/api/pipeline/defaults')]);
  S.sources = srcs.items;
  const d = def.defaults, opt = def.options;

  $('#topbar-actions').innerHTML = `<button class="btn btn-sm" id="btn-newsrc">＋ 新建数据源</button>`;

  c.innerHTML = `
    <div class="card">
      <div class="card-head"><div><div class="card-title">数据源配置</div>
        <div class="card-sub">支持本地 CSV、HTTP 直链 CSV、SQLite 表三类。配置以 JSON 持久化在 <code>config/datasources.json</code>，前端与独立导入工具共用同一份配置。</div></div></div>
      <div class="table-wrap"><table><thead><tr>
        <th>ID</th><th>名称</th><th>类型</th><th>位置 / URI</th><th>编码</th><th>状态</th><th>最近导入</th><th></th>
      </tr></thead><tbody id="src-body"></tbody></table></div>
    </div>

    <div class="grid g2-21">
      <div class="card">
        <div class="card-head"><div><div class="card-title">流水线参数</div>
          <div class="card-sub">参数会完整写入数据版本记录，保证任何一次运行都可复现</div></div></div>

        <div class="sec-title">① 数据导入</div>
        <div class="field"><label>数据源</label><select id="p-source"></select></div>
        <div class="field"><label>分块大小 <span class="hint">chunk_size，行</span></label>
          <input type="number" id="p-chunk" value="${d.chunk_size}" min="10000" step="10000"></div>

        <div class="sec-title">② 缺失值处理</div>
        <div class="field"><label>全局策略</label>
          <select id="p-missing">${opt.missing_strategy.map(x => `<option value="${x}" ${x === d.missing_strategy ? 'selected' : ''}>${{ drop: '删除含缺失的行', mean: '均值填充', median: '中位数填充', mode: '众数填充', constant: '常数填充(0/未知)', ffill: '前向填充' }[x]}</option>`).join('')}</select></div>
        <div class="field"><label>高缺失率列剔除阈值 <span class="hint">缺失率高于该值的特征列整列删除</span></label>
          <input type="number" id="p-dropt" value="${d.drop_threshold}" min="0.1" max="1" step="0.05"></div>

        <div class="sec-title">③ 异常值处理</div>
        <div class="field"><label>检测策略</label>
          <select id="p-outlier">${opt.outlier_strategy.map(x => `<option value="${x}" ${x === d.outlier_strategy ? 'selected' : ''}>${{ iqr: 'IQR 四分位距', zscore: 'Z-Score 3σ', range: '业务区间校验', none: '不处理' }[x]}</option>`).join('')}</select></div>
        <div class="field"><label>处置动作</label>
          <select id="p-oaction">${(opt.outlier_action || ['winsorize', 'clip_median', 'drop']).map(x => `<option value="${x}" ${x === d.outlier_action ? 'selected' : ''}>${{ winsorize: 'winsorize 封顶到边界（推荐）', clip_median: 'clip_median 置为中位数', drop: 'drop 删除越界记录' }[x]}</option>`).join('')}</select>
          <div class="hint" style="margin-top:4px;font-weight:400">仅对业务列（单价/总价/面积/小区均价/建成年份/梯户比）检测；重尾计数量（挂牌天数、关注人数）不参与，避免被压成常数。</div></div>
        <div class="field"><label>IQR 倍数 k</label><input type="number" id="p-iqrk" value="${d.iqr_k}" min="0.5" max="5" step="0.5"></div>

        <div class="sec-title">④ 数据划分</div>
        <div class="field"><label>划分协议</label>
          <select id="p-split">
            <option value="time" ${d.split_protocol === 'time' ? 'selected' : ''}>按成交时间切分（推荐，贴近真实预测）</option>
            <option value="random" ${d.split_protocol === 'random' ? 'selected' : ''}>随机切分</option>
          </select></div>
        <div class="range-row">
          <div class="field"><label>测试集比例</label><input type="number" id="p-test" value="${d.test_size}" min="0.05" max="0.5" step="0.05"></div>
          <div class="field"><label>随机种子</label><input type="number" id="p-seed" value="${d.seed}"></div>
        </div>

        <div class="sec-title">⑤ 存储</div>
        <div class="flex wrap" style="margin-bottom:12px">
          <label class="check"><input type="checkbox" id="p-sqlite" checked> 写入 SQLite 明细表</label>
          <label class="check"><input type="checkbox" id="p-parquet" checked> 保存 Parquet 列存快照</label>
        </div>
        <div class="field"><label>写入模式</label>
          <select id="p-mode"><option value="replace">replace 全量重建</option><option value="upsert">upsert 按 house_id 增量合并</option></select></div>

        <div class="flex" style="margin-top:16px;gap:9px">
          <button class="btn btn-primary" id="btn-run" style="flex:1">▶ 运行流水线</button>
          <button class="btn" id="btn-init" title="一键完成全流程">⚡ 全流程</button>
        </div>
      </div>

      <div class="card">
        <div class="card-head"><div class="card-title">运行日志</div>
          <div class="flex"><span id="job-status"></span></div></div>
        <div id="job-log" class="log" style="max-height:520px">等待运行 …</div>
        <div id="job-result" style="margin-top:14px"></div>
      </div>
    </div>

    <div class="card"><div class="card-head"><div><div class="card-title">独立数据导入 / 更新工具</div>
      <div class="card-sub">命令行工具，适合定时批处理与增量更新</div></div></div>
      <pre class="doc" style="max-height:none"># 下载 URL 数据源到本地 raw 目录
python tools/etl.py download --source lianjia_bj_raw

# 运行完整数据流水线（可覆盖任意参数）
python tools/etl.py run --source lianjia_bj_raw --missing median --outlier iqr --split time

# 增量更新：把新批次成交数据合并进明细表
python tools/etl.py update --source lianjia_bj_raw --csv data/raw/new_batch.csv

# 查看已有数据版本
python tools/etl.py versions

# 大文件切分为分片（便于分批/并行处理）
python tools/etl.py split --source lianjia_bj_raw --rows 100000</pre>
    </div>
  `;

  // 数据源行
  const body = $('#src-body');
  body.innerHTML = srcs.items.map(s => `<tr>
      <td class="mono">${esc(s.source_id)}</td>
      <td>${esc(s.name)}</td>
      <td><span class="tag ${s.source_type === 'url_csv' ? 'tag-purple' : 'tag-blue'}">${esc(s.source_type)}</span></td>
      <td class="mono" style="max-width:380px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(s.uri)}">${esc(s.uri)}</td>
      <td class="mono">${esc(s.encoding)}</td>
      <td>${s.exists ? '<span class="tag tag-green">文件就绪</span>' : '<span class="tag tag-gold">待下载</span>'}</td>
      <td class="muted">${esc(s.last_run_at || '—')}${s.last_run_rows ? ` · ${num(s.last_run_rows)} 行` : ''}</td>
      <td class="flex">
        <button class="btn btn-sm" data-probe="${esc(s.source_id)}">探测</button>
        ${s.source_type === 'url_csv' ? `<button class="btn btn-sm" data-dl="${esc(s.source_id)}">下载</button>` : ''}
        <button class="btn btn-sm btn-ghost" data-edit="${esc(s.source_id)}">编辑</button>
        <button class="btn btn-sm btn-ghost" data-del="${esc(s.source_id)}">删除</button>
      </td></tr>`).join('');

  const sel = $('#p-source');
  sel.innerHTML = srcs.items.map(s => `<option value="${esc(s.source_id)}" ${s.source_id === 'lianjia_bj_raw' ? 'selected' : ''}>${esc(s.name)}</option>`).join('');

  body.addEventListener('click', async e => {
    const probe = e.target.dataset.probe, dl = e.target.dataset.dl, del = e.target.dataset.del, ed = e.target.dataset.edit;
    if (probe) {
      try { const r = await POST(`/api/sources/${probe}/probe`); toast(`编码 ${r.encoding || '—'} · ${r.size_mb || 0} MB`, 'ok'); } catch (err) { toast(err.message, 'err'); }
    }
    if (dl) { toast('开始下载 …'); try { const r = await POST(`/api/sources/${dl}/download`); toast(`下载完成 ${r.size_mb} MB`, 'ok'); go('pipeline'); } catch (err) { toast(err.message, 'err'); } }
    if (del) { if (confirm('确认删除该数据源配置？')) { await api('DELETE', `/api/sources/${del}`); toast('已删除', 'ok'); go('pipeline'); } }
    if (ed) { editSource(srcs.items.find(x => x.source_id === ed)); }
  });

  $('#btn-newsrc').onclick = () => editSource(null);
  $('#btn-run').onclick = () => runPipeline(false);
  $('#btn-init').onclick = () => runPipeline(true);
}

function editSource(s) {
  const src = s || { source_id: '', name: '', source_type: 'local_csv', uri: '', encoding: 'auto', delimiter: ',', city: '北京', description: '', license: '', attribution: '' };
  const m = modal(s ? '编辑数据源' : '新建数据源', `
    <div class="form-grid">
      <div class="field"><label>数据源 ID</label><input id="sim-id" value="${esc(src.source_id)}" ${s ? 'disabled' : ''} placeholder="留空自动生成"></div>
      <div class="field"><label>名称</label><input id="src-name" value="${esc(src.name)}"></div>
      <div class="field"><label>类型</label><select id="src-type">
        <option value="local_csv" ${src.source_type === 'local_csv' ? 'selected' : ''}>local_csv 本地文件</option>
        <option value="url_csv" ${src.source_type === 'url_csv' ? 'selected' : ''}>url_csv 远程直链</option>
        <option value="sqlite" ${src.source_type === 'sqlite' ? 'selected' : ''}>sqlite 数据表</option></select></div>
      <div class="field"><label>编码</label><input id="src-enc" value="${esc(src.encoding)}" placeholder="auto / gb18030 / utf-8"></div>
      <div class="field"><label>分隔符</label><input id="src-delim" value="${esc(src.delimiter)}"></div>
      <div class="field"><label>城市</label><input id="src-city" value="${esc(src.city)}"></div>
    </div>
    <div class="field"><label>位置 / URI</label><input id="src-uri" value="${esc(src.uri)}" placeholder="本地路径 或 https://... 或 sqlite 表名"></div>
    <div class="field"><label>描述</label><textarea id="src-desc">${esc(src.description)}</textarea></div>
    <div class="form-grid">
      <div class="field"><label>许可协议</label><input id="src-lic" value="${esc(src.license)}"></div>
      <div class="field"><label>来源标注</label><input id="src-attr" value="${esc(src.attribution)}"></div>
    </div>
    <div class="flex" style="justify-content:flex-end"><button class="btn btn-primary" id="src-save">保存</button></div>
  `);
  $('#src-save', m).onclick = async () => {
    try {
      await POST('/api/sources', {
        source_id: $('#sim-id', m).value || undefined, name: $('#src-name', m).value,
        source_type: $('#src-type', m).value, uri: $('#src-uri', m).value, encoding: $('#src-enc', m).value,
        delimiter: $('#src-delim', m).value, city: $('#src-city', m).value, description: $('#src-desc', m).value,
        license: $('#src-lic', m).value, attribution: $('#src-attr', m).value,
      });
      m.remove(); toast('已保存', 'ok'); go('pipeline');
    } catch (e) { toast(e.message, 'err'); }
  };
}

async function runPipeline(full) {
  const params = {
    chunk_size: +$('#p-chunk').value,
    missing_strategy: $('#p-missing').value,
    drop_threshold: +$('#p-dropt').value,
    outlier_strategy: $('#p-outlier').value,
    outlier_action: $('#p-oaction').value,
    iqr_k: +$('#p-iqrk').value,
    split_protocol: $('#p-split').value,
    test_size: +$('#p-test').value,
    seed: +$('#p-seed').value,
    store_sqlite: $('#p-sqlite').checked,
    save_parquet: $('#p-parquet').checked,
    store_mode: $('#p-mode').value,
  };
  const body = { source_id: $('#p-source').value, params };
  if (full) { body.params.split_protocol = params.split_protocol; }

  $('#btn-run').disabled = true; $('#btn-init').disabled = true;
  const log = $('#job-log');
  log.innerHTML = `<span class="dim">[${new Date().toLocaleTimeString()}] 提交任务 …</span>\n`;
  $('#job-result').innerHTML = '';
  try {
    const r = await POST('/api/pipeline/run', body);
    pollJob(r.job_id, log, async (res) => {
      renderJobResult(res);
      toast('数据流水线完成', 'ok');
      $('#btn-run').disabled = false; $('#btn-init').disabled = false;
      refreshHealth();
    }, () => { $('#btn-run').disabled = false; $('#btn-init').disabled = false; });
  } catch (e) {
    toast(e.message, 'err'); $('#btn-run').disabled = false; $('#btn-init').disabled = false;
  }
}

function pollJob(jobId, logEl, onDone, onFail) {
  const t0 = Date.now();
  const tick = async () => {
    try {
      const j = await GET(`/api/jobs/${jobId}`);
      const secs = ((Date.now() - t0) / 1000).toFixed(0);
      if (j.status === 'running' || j.status === 'queued') {
        logEl.innerHTML = `<span class="dim">[${new Date().toLocaleTimeString()}] 任务 ${jobId}</span>\n` +
          `<span class="dim">状态: ${j.status} · 已运行 ${secs}s</span>\n\n` +
          `正在执行：导入 → 清洗 → 特征 → 划分 → 存储 → 预聚合 …\n` +
          `<span class="dim">（31 万行数据的完整流水线通常需要 1–3 分钟，请稍候）</span>`;
        setTimeout(tick, 1500);
      } else if (j.status === 'success') {
        logEl.innerHTML = `<span class="dim">[${j.started_at}] 任务启动</span>\n` +
          `<span class="ok">[${j.finished_at}] ✓ 完成，耗时 ${j.seconds}s</span>\n\n` +
          formatJobLog(j.result);
        onDone && onDone(j.result);
      } else {
        logEl.innerHTML = `<span class="er">[${j.finished_at}] ✕ 失败：${esc(j.error)}</span>\n\n` +
          `<span class="dim">${esc((j.traceback || '').slice(-1500))}</span>`;
        toast(j.error, 'err', 6000);
        onFail && onFail();
      }
    } catch (e) { logEl.innerHTML += `\n<span class="er">轮询失败: ${esc(e.message)}</span>`; onFail && onFail(); }
  };
  tick();
}

function formatJobLog(res) {
  const v = res.version || res;
  if (!v || !v.stages) return `<span class="dim">${esc(JSON.stringify(res).slice(0, 600))}</span>`;
  let out = `<span class="ok">版本号 ${esc(v.version_id)}</span>\n`;
  out += `来源: ${esc(v.source_name)}  |  许可: ${esc(v.source_license || '—')}\n`;
  if (v.ingest_report) out += `编码 ${esc(v.ingest_report.encoding)} · 分块 ${v.ingest_report.chunks} 次 · 原始 ${num(v.raw_rows)} 行\n`;
  out += '\n' + v.stages.map(s =>
    `  ${String(s.stage).padEnd(10)} ${String(num(s.rows_in)).padStart(9)} → ${String(num(s.rows_out)).padStart(9)} 行` +
    `  (剔除 ${num(s.dropped)})  ${s.seconds}s  ${s.notes || ''}`
  ).join('\n');
  if (v.target_stats) out += `\n\n最终 ${num(v.final_rows)} 行 / ${v.final_columns} 列；单价均值 ${num(v.target_stats.mean)} 元/㎡，中位数 ${num(v.target_stats.median)}`;
  if (v.checksum) out += `\n数据校验和(SHA-256 前 16 位): ${esc(v.checksum.slice(0, 16))}`;
  return out;
}

function renderJobResult(res) {
  const v = res.version || res;
  if (!v || !v.stages) return;
  $('#job-result').innerHTML = `
    <div class="alert alert-ok"><span>✓</span><div><b>数据版本 ${esc(v.version_id)}</b> 已登记，最终 ${num(v.final_rows)} 行可建模数据。</div></div>
    <div class="grid g4">
      ${kpi('原始行数', num(v.raw_rows), '条', '', '')}
      ${kpi('清洗后行数', num(v.final_rows), '条', `剔除 ${num((v.raw_rows || 0) - (v.final_rows || 0))} 条`, 'k-green')}
      ${kpi('特征列数', num(v.final_columns), '列', `建模特征 ${num((v.stages.find(s => s.stage === 'features') || {}).params?.n_features || 0)} 个`, 'k-purple')}
      ${kpi('总耗时', num(v.seconds_total, 1), '秒', `${v.stages.length} 个阶段`, 'k-gold')}
    </div>
    <div class="flex" style="margin-top:14px">
      <button class="btn btn-sm btn-primary" onclick="go('versions')">查看版本记录 →</button>
      <button class="btn btn-sm" onclick="go('training')">去训练模型 →</button>
    </div>`;
}

/* ============================================================ 3. 数据版本 */
async function renderVersions() {
  const c = $('#content');
  const r = await GET('/api/versions');
  S.versions = r.items;
  if (!r.items.length) { c.innerHTML = emptyState('暂无数据版本', '运行一次数据流水线后，这里会记录每次运行的参数、行数与校验和。', `<button class="btn btn-primary" onclick="go('pipeline')">前往数据导入 →</button>`); return; }

  c.innerHTML = `
    <div class="card">
      <div class="card-head"><div><div class="card-title">数据版本登记簿</div>
        <div class="card-sub">每个版本 = 一次完整的流水线运行。记录来源指纹、各阶段参数与行数变化、输出产物校验和，用于复现与回溯。</div></div>
        <button class="btn btn-sm" onclick="go('pipeline')">＋ 新版本</button></div>
      <div class="table-wrap"><table><thead><tr>
        <th></th><th>版本号</th><th>创建时间</th><th>数据源</th><th class="num">原始行</th><th class="num">最终行</th>
        <th class="num">剔除率</th><th>状态</th><th>校验和</th><th></th></tr></thead><tbody id="vbody"></tbody></table></div>
    </div>
    <div class="card">
      <div class="card-head"><div class="card-title">版本对比</div>
        <div class="card-sub">选择两个版本，对比规模、清洗损失与目标变量分布</div></div>
      <div class="flex wrap" style="margin-bottom:12px">
        <select id="cmp-a" style="max-width:300px">${S.versions.map(v => `<option value="${v.version_id}">${v.version_id}</option>`).join('')}</select>
        <span class="muted">对比</span>
        <select id="cmp-b" style="max-width:300px">${S.versions.map((v, i) => `<option value="${v.version_id}" ${i === 1 ? 'selected' : ''}>${v.version_id}</option>`).join('')}</select>
        <button class="btn btn-sm btn-primary" id="btn-cmp">开始对比</button>
      </div>
      <div id="cmp-out"></div>
    </div>`;

  $('#vbody').innerHTML = r.items.map(v => `<tr>
    <td>${v.active ? '<span class="tag tag-green">当前</span>' : ''}</td>
    <td class="mono">${esc(v.version_id)}</td>
    <td class="muted">${esc(v.created_at || '')}</td>
    <td>${esc(v.source_name || '')}</td>
    <td class="num">${num(v.raw_rows)}</td>
    <td class="num">${num(v.final_rows)}</td>
    <td class="num">${v.raw_rows ? ((1 - v.final_rows / v.raw_rows) * 100).toFixed(2) + '%' : '—'}</td>
    <td>${v.status === 'success' ? '<span class="tag tag-green">成功</span>' : '<span class="tag tag-red">失败</span>'}</td>
    <td class="mono" title="${esc(v.checksum || '')}">${esc((v.checksum || '—').slice(0, 12))}</td>
    <td class="flex">
      ${v.active ? '' : `<button class="btn btn-sm" data-act="${esc(v.version_id)}">激活</button>`}
      <button class="btn btn-sm btn-ghost" data-view="${esc(v.version_id)}">详情</button>
      <button class="btn btn-sm btn-ghost" data-del="${esc(v.version_id)}">删除</button>
    </td></tr>`).join('');

  $('#vbody').addEventListener('click', async e => {
    const a = e.target.dataset.act, v = e.target.dataset.view, d = e.target.dataset.del;
    if (a) { await POST(`/api/versions/${a}/activate`); toast('已激活为当前版本', 'ok'); go('versions'); }
    if (v) { const r2 = await GET(`/api/versions/${v}`); modal('数据版本详情 ' + v, versionDetail(r2.version), true); }
    if (d) { if (confirm('删除该版本记录？（不会删除已训练模型）')) { await api('DELETE', `/api/versions/${d}`); go('versions'); } }
  });

  $('#btn-cmp').onclick = async () => {
    try {
      const out = await GET(`/api/analytics/compare?a=${$('#cmp-a').value}&b=${$('#cmp-b').value}`);
      $('#cmp-out').innerHTML = compareTable(out.a, out.b);
    } catch (e) { toast(e.message, 'err'); }
  };
}

function versionDetail(v) {
  const st = (v.stages || []).map(s => `<tr><td><span class="tag tag-blue">${esc(s.stage)}</span></td>
    <td class="num">${num(s.rows_in)}</td><td class="num">${num(s.rows_out)}</td><td class="num">${num(s.dropped)}</td>
    <td class="num">${s.seconds}s</td><td class="muted">${esc(s.notes || '')}</td></tr>`).join('');
  const cr = v.clean_report || {}, miss = cr.missing || {}, outl = cr.outlier || {};
  return `
    <div class="grid g4">
      ${kpi('原始行数', num(v.raw_rows), '', '', '')}
      ${kpi('最终行数', num(v.final_rows), '', '', 'k-green')}
      ${kpi('总耗时', num(v.seconds_total, 1), 's', '', 'k-gold')}
      ${kpi('校验和', esc((v.checksum || '—').slice(0, 10)), '', 'SHA-256', 'k-purple')}
    </div>
    <div class="sec-title">阶段执行记录</div>
    <div class="table-wrap"><table><thead><tr><th>阶段</th><th class="num">输入行</th><th class="num">输出行</th><th class="num">剔除</th><th class="num">耗时</th><th>备注</th></tr></thead><tbody>${st}</tbody></table></div>
    <div class="grid g2" style="margin-top:16px">
      <div><div class="sec-title">缺失值处理</div>
        <pre class="doc" style="max-height:220px">${esc(JSON.stringify(miss, null, 2))}</pre></div>
      <div><div class="sec-title">异常值处理</div>
        <pre class="doc" style="max-height:220px">${esc(JSON.stringify({ strategy: outl.strategy, outliers_detected: outl.outliers_detected, affected_columns: Object.keys(outl.affected_columns || {}) }, null, 2))}</pre></div>
    </div>
    <div class="sec-title">来源与许可</div>
    <pre class="doc" style="max-height:160px">${esc(JSON.stringify({ source: v.source_name, uri: v.source_uri, license: v.source_license, attribution: v.source_attribution }, null, 2))}</pre>`;
}

function compareTable(a, b) {
  const rows = [['原始行数', a.raw_rows, b.raw_rows], ['最终行数', a.final_rows, b.final_rows],
  ['最终列数', a.final_columns, b.final_columns], ['单价均值', (a.target_stats || {}).mean, (b.target_stats || {}).mean],
  ['单价中位数', (a.target_stats || {}).median, (b.target_stats || {}).median],
  ['总耗时(s)', a.seconds_total, b.seconds_total]];
  return `<div class="table-wrap"><table><thead><tr><th>指标</th><th class="num">${esc(a.version_id)}</th><th class="num">${esc(b.version_id)}</th><th class="num">差异</th></tr></thead><tbody>
    ${rows.map(([k, x, y]) => `<tr><td>${k}</td><td class="num">${num(x, 1)}</td><td class="num">${num(y, 1)}</td>
      <td class="num" style="color:${(Number(y) - Number(x)) >= 0 ? '#d93a34' : '#178f57'}">${x != null && y != null ? ((Number(y) - Number(x)) >= 0 ? '+' : '') + num(Number(y) - Number(x), 1) : '—'}</td></tr>`).join('')}
  </tbody></table></div>`;
}

/* ============================================================ 4. 清洗与质量 */
async function renderQuality() {
  const c = $('#content');
  const vs = await GET('/api/versions');
  if (!vs.items.length) { c.innerHTML = emptyState('暂无数据', '请先运行数据流水线。'); return; }
  const pick = vs.items.find(v => v.active) || vs.items[0];
  c.innerHTML = `
    <div class="card"><div class="card-head"><div class="card-title">数据质量报告</div>
      <select id="q-ver" style="max-width:320px">${vs.items.map(v => `<option value="${v.version_id}" ${v.version_id === pick.version_id ? 'selected' : ''}>${v.version_id}</option>`).join('')}</select>
    </div><div id="q-body"><div class="skeleton" style="height:200px"></div></div></div>`;
  const load = async (vid) => {
    const q = await GET(`/api/versions/${vid}/quality`);
    $('#q-body').innerHTML = qualityHtml(q);
    renderQualityCharts(q);
  };
  $('#q-ver').onchange = e => load(e.target.value);
  await load(pick.version_id);
}

function qualityHtml(q) {
  const cr = q.clean_report || {}, f = cr.filter || {}, m = cr.missing || {}, o = cr.outlier || {};
  const prof = (q.profile_raw && q.profile_raw.profile) || [];
  const ft = q.feature_report || {};
  return `
    <div class="grid g4" style="margin-bottom:16px">
      ${kpi('原始记录', num(cr.rows_in), '条', '', '')}
      ${kpi('清洗后记录', num(cr.rows_out), '条', `剔除 ${num(cr.rows_dropped_total)} 条（${pct(cr.rows_in ? cr.rows_dropped_total / cr.rows_in * 100 : 0)}）`, 'k-green')}
      ${kpi('填充单元格', num(m.filled_cells), '个', `高缺失列剔除 ${(m.columns_dropped_high_missing || []).length} 个`, 'k-gold')}
      ${kpi('修复异常值', num(o.outliers_detected), '个', `策略 ${o.strategy || '—'}`, 'k-red')}
    </div>

    <div class="grid g2">
      <div class="card"><div class="card-head"><div class="card-title">记录过滤明细</div>
        <div class="card-sub">每条规则剔除的记录数</div></div>
        <div id="ch-filter" class="chart chart-sm"></div></div>
      <div class="card"><div class="card-head"><div class="card-title">异常值检出分布</div>
        <div class="card-sub">按列统计被判定为异常并修复的记录数与占比</div></div>
        <div id="ch-outlier" class="chart chart-sm"></div></div>
    </div>

    <div class="grid g2">
      <div class="card"><div class="card-head"><div class="card-title">缺失值处理</div></div>
        <div class="table-wrap" style="max-height:280px;overflow:auto"><table><thead><tr><th>列</th><th>策略</th><th class="num">填充数</th><th class="num">原缺失率</th></tr></thead>
        <tbody>${Object.entries(m.fill_detail || {}).map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td><span class="tag tag-blue">${esc(v.strategy)}</span></td><td class="num">${num(v.filled)}</td><td class="num">${pct(v.missing_rate * 100)}</td></tr>`).join('') || '<tr><td colspan="4" class="muted" style="text-align:center;padding:22px">本次运行无缺失值需要填充</td></tr>'}</tbody></table></div>
        ${(m.columns_dropped_high_missing || []).length ? `<div class="alert alert-warn" style="margin-top:12px"><span>!</span><div>高缺失率被整列剔除：<b>${(m.columns_dropped_high_missing || []).map(x => `${esc(x.column)} (${pct(x.missing_rate * 100)})`).join('、')}</b></div></div>` : ''}
      </div>
      <div class="card"><div class="card-head"><div class="card-title">异常值处理明细</div></div>
        <div class="table-wrap" style="max-height:280px;overflow:auto"><table><thead><tr><th>列</th><th class="num">检出</th><th class="num">占比</th><th class="num">下界</th><th class="num">上界</th></tr></thead>
        <tbody>${Object.entries(o.affected_columns || {}).map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td class="num">${num(v.detected)}</td><td class="num">${pct(v.rate * 100, 3)}</td><td class="num">${num(v.lower_bound, 0)}</td><td class="num">${num(v.upper_bound, 0)}</td></tr>`).join('') || '<tr><td colspan="5" class="muted" style="text-align:center;padding:22px">未检出异常值</td></tr>'}</tbody></table></div>
      </div>
    </div>

    <div class="card"><div class="card-head"><div class="card-title">特征工程结果</div>
      <div class="card-sub">构造出的建模特征（已通过防泄漏校验）</div></div>
      <div class="flex wrap">${(ft.model_features || []).map(x => `<span class="tag tag-blue" style="margin:3px">${esc(x)}</span>`).join('')}</div>
      <div class="grid g3" style="margin-top:14px">
        <div>${kpi('特征总数', num(ft.n_features), '个', '', 'k-purple')}</div>
        <div>${kpi('防泄漏校验', ft.forbidden_check === 'passed' ? '通过' : '—', '', 'unit_price / total_price 派生量已排除', 'k-green')}</div>
        <div>${kpi('划分协议', esc((q.split_meta || {}).protocol || '—'), '', `训练 ${num((q.split_meta || {}).train_rows)} / 测试 ${num((q.split_meta || {}).test_rows)}`, 'k-gold')}</div>
      </div>
      ${(q.split_meta || {}).train_period ? `<div class="alert alert-info" style="margin-top:14px"><span>i</span><div>时间划分：训练集 <b>${esc(q.split_meta.train_period[0].slice(0, 10))} ~ ${esc(q.split_meta.train_period[1].slice(0, 10))}</b>，测试集 <b>${esc(q.split_meta.test_period[0].slice(0, 10))} ~ ${esc(q.split_meta.test_period[1].slice(0, 10))}</b>。用早期成交预测后期成交，更贴近真实预测场景。</div></div>` : ''}
    </div>

    <div class="card"><div class="card-head"><div><div class="card-title">原始数据画像</div>
      <div class="card-sub">导入阶段的列级统计（dtype / 缺失 / 唯一值）</div></div></div>
      <div class="table-wrap" style="max-height:420px;overflow:auto"><table><thead><tr>
        <th>列名</th><th>类型</th><th class="num">非空</th><th class="num">缺失</th><th class="num">缺失率</th><th class="num">唯一值</th><th>样例</th>
      </tr></thead><tbody>${prof.map(p => `<tr>
        <td class="mono">${esc(p.column)}</td><td class="muted">${esc(p.dtype)}</td>
        <td class="num">${num(p.non_null)}</td><td class="num">${num(p.missing)}</td>
        <td class="num" style="color:${p.missing_rate > 0.3 ? '#d93a34' : p.missing_rate > 0.05 ? '#c98a2e' : '#178f57'}">${pct(p.missing_rate * 100, 2)}</td>
        <td class="num">${num(p.n_unique)}</td><td class="muted" style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(p.sample)}</td>
      </tr>`).join('')}</tbody></table></div>
    </div>`;
}

function renderQualityCharts(q) {
  const f = (q.clean_report || {}).filter || {};
  const labels = {
    drop_missing_required: '必填字段缺失', drop_exact_duplicate: '完全重复记录',
    drop_duplicate_house_id: '同一房源重复', drop_nonpositive_target: '单价非正',
    drop_target_out_of_business_range: '单价超出业务区间',
  };
  const items = Object.entries(f).filter(([k, v]) => typeof v === 'number' && k.startsWith('drop_'));
  ec('ch-filter', {
    tooltip: { ...TIP, formatter: p => `${p[0].name}: ${num(p[0].value)} 条` },
    grid: { left: 140, right: 52, top: 16, bottom: 22 },
    xAxis: { type: 'value', ...AXIS },
    yAxis: { type: 'category', data: items.map(([k]) => labels[k] || k.replace('drop_', '')), ...AXIS },
    series: [{ type: 'bar', data: items.map(([, v]) => v), itemStyle: { color: '#c98a2e', borderRadius: [0, 4, 4, 0] }, barMaxWidth: 22, label: { show: true, position: 'right', fontSize: 10, color: '#55637a', formatter: p => num(p.value) } }],
  });

  const oc = (q.clean_report || {}).outlier || {};
  const entries = Object.entries(oc.affected_columns || {}).sort((a, b) => b[1].detected - a[1].detected).slice(0, 10);
  ec('ch-outlier', {
    tooltip: { ...TIP, formatter: p => `${p[0].name}<br/>检出 ${num(p[0].value)} 条` },
    grid: { left: 118, right: 46, top: 16, bottom: 22 },
    xAxis: { type: 'value', ...AXIS },
    yAxis: { type: 'category', data: entries.map(([k]) => k), ...AXIS },
    series: [{ type: 'bar', data: entries.map(([, v]) => v.detected), itemStyle: { color: '#d93a34', borderRadius: [0, 4, 4, 0] }, barMaxWidth: 22, label: { show: true, position: 'right', fontSize: 10, color: '#55637a', formatter: p => num(p.value) } }],
  });
}

/* ============================================================ 5. 房源明细 */
let listingPage = 0;
async function renderListings() {
  const c = $('#content');
  c.innerHTML = `
    <div class="card"><div class="card-head"><div><div class="card-title">条件查询</div>
      <div class="card-sub">直接查询 SQLite 明细表 <code>house_detail</code></div></div></div>
      <div class="form-grid">
        <div class="field"><label>区域</label><select id="l-district"><option value="">全部</option></select></div>
        <div class="field"><label>最低单价 元/㎡</label><input type="number" id="l-minp" placeholder="不限"></div>
        <div class="field"><label>最高单价 元/㎡</label><input type="number" id="l-maxp" placeholder="不限"></div>
        <div class="field"><label>最小面积 ㎡</label><input type="number" id="l-mina" placeholder="不限"></div>
        <div class="field"><label>最大面积 ㎡</label><input type="number" id="l-maxa" placeholder="不限"></div>
        <div class="field"><label>最少房间数</label><input type="number" id="l-rooms" placeholder="不限"></div>
      </div>
      <div class="flex"><button class="btn btn-primary" id="l-search">查询</button>
        <button class="btn" id="l-reset">重置</button><span class="muted" id="l-count"></span></div>
    </div>
    <div class="card"><div class="table-wrap" style="max-height:600px;overflow:auto" id="l-table"></div>
      <div class="flex" style="justify-content:center;margin-top:14px">
        <button class="btn btn-sm" id="l-prev">← 上一页</button><span class="muted" id="l-page"></span>
        <button class="btn btn-sm" id="l-next">下一页 →</button></div>
    </div>`;

  try {
    const agg = await GET('/api/analytics/district');
    $('#l-district').innerHTML = '<option value="">全部</option>' + (agg.items || []).map(x => `<option>${esc(x.district_name)}</option>`).join('');
  } catch (e) { }

  const doSearch = async (page) => {
    listingPage = page;
    const q = new URLSearchParams({ limit: 25, offset: page * 25 });
    const d = $('#l-district').value, mp = $('#l-minp').value, xp = $('#l-maxp').value,
      ma = $('#l-mina').value, xa = $('#l-maxa').value, rm = $('#l-rooms').value;
    if (d) q.set('district', d);
    if (mp) q.set('min_price', mp); if (xp) q.set('max_price', xp);
    if (ma) q.set('min_area', ma); if (xa) q.set('max_area', xa);
    if (rm) q.set('rooms', rm);
    const r = await GET('/api/listings?' + q.toString());
    $('#l-count').textContent = `共 ${num(r.total)} 条`;
    $('#l-page').textContent = `第 ${page + 1} / ${Math.max(1, Math.ceil(r.total / 25))} 页`;
    $('#l-table').innerHTML = `<table><thead><tr>
      <th>房源 ID</th><th>区域</th><th class="num">面积㎡</th><th class="num">户型</th><th class="num">单价元/㎡</th>
      <th class="num">总价万</th><th>楼层</th><th>建成年</th><th>装修</th><th>电梯</th><th>地铁</th>
      <th class="num">关注</th><th class="num">挂牌天</th><th>成交时间</th></tr></thead>
      <tbody>${r.items.map(x => `<tr>
        <td class="mono">${esc(x.house_id)}</td><td>${esc(x.district_name)}</td>
        <td class="num">${num(x.area, 1)}</td><td class="num">${num(x.room_total)}</td>
        <td class="num" style="color:#d93a34;font-weight:600">${num(x.unit_price)}</td>
        <td class="num">${num(x.total_price, 1)}</td><td>${esc(x.floor_category || '')}</td>
        <td class="num">${x.build_year ? num(x.build_year) : '—'}</td><td>${esc(x.renovation_name || '')}</td>
        <td>${esc(x.elevator_name || '')}</td><td>${esc(x.subway_name || '')}</td>
        <td class="num">${num(x.followers)}</td><td class="num">${num(x.days_on_market)}</td>
        <td class="muted">${esc((x.trade_time || '').slice(0, 10))}</td></tr>`).join('')}</tbody></table>`;
  };
  $('#l-search').onclick = () => doSearch(0);
  $('#l-reset').onclick = () => { ['l-minp', 'l-maxp', 'l-mina', 'l-maxa', 'l-rooms'].forEach(i => $('#' + i).value = ''); $('#l-district').value = ''; doSearch(0); };
  $('#l-prev').onclick = () => { if (listingPage > 0) doSearch(listingPage - 1); };
  $('#l-next').onclick = () => doSearch(listingPage + 1);
  await doSearch(0);
}

/* ============================================================ 6. 算法与参数配置 */
async function renderAlgorithms() {
  const c = $('#content');
  const r = await GET('/api/algorithms');
  S.algorithms = r.items; S.tasks = r.tasks; S.featureProfiles = r.feature_profiles || [];

  const price = r.items.filter(a => a.task === 'price_regression');
  const rec = r.items.filter(a => a.task === 'recommendation');

  c.innerHTML = `
    <div class="alert alert-info"><span>i</span><div>
      共注册 <b>${r.items.length}</b> 个算法，分属 2 个任务方向。每个算法都声明了<b>输入契约、输出结果、主要参数（类型/默认值/取值范围）与评价方法</b>；
      参数表单在各功能页自动渲染，配置会随训练/推荐请求一并提交。
    </div></div>
    ${algoSection('价格预测（回归）', 'price_regression', price)}
    ${algoSection('房源推荐', 'recommendation', rec)}`;

  c.addEventListener('click', e => {
    const id = e.target.dataset.algo;
    if (id) openAlgoDetail(id);
    const train = e.target.dataset.train;
    if (train) { sigTrain = { algo_id: train }; go('training'); }
    const use = e.target.dataset.use;
    if (use) { go('recommend'); setTimeout(() => { const s = $('#r-algo'); if (s) { s.value = use; s.dispatchEvent(new Event('change')); } }, 600); }
  });
}

function algoSection(title, task, list) {
  return `<div class="card"><div class="card-head"><div><div class="card-title">${esc(title)}</div>
    <div class="card-sub">任务标识 <code>${esc(task)}</code> · 共 ${list.length} 个算法</div></div></div>
    <div class="grid g2">${list.map(a => algoCard(a)).join('')}</div></div>`;
}

function algoCard(a) {
  const cls = { '线性模型': 'tag-blue', '树模型': 'tag-green', '基线': 'tag-gold', '内容/相似度': 'tag-purple', '协同过滤': 'tag-red', '混合/召回排序': 'tag-gold', '相似度计算': 'tag-blue' }[a.category] || '';
  const action = a.task === 'price_regression'
    ? `<button class="btn btn-sm btn-primary" data-train="${a.id}">用此算法训练 →</button>`
    : `<button class="btn btn-sm" data-use="${a.id}">去推荐页使用 →</button>`;
  return `<div class="kpi" style="padding:15px 17px">
    <div class="flex between" style="align-items:flex-start">
      <div><div style="font-weight:680;font-size:14px">${esc(a.name)}</div>
        <div class="flex" style="gap:5px;margin-top:6px">
          <span class="tag ${cls}">${esc(a.category)}</span>
          ${a.interpretable ? '<span class="tag tag-green">可解释</span>' : '<span class="tag">需解释工具</span>'}
          <span class="tag">${a.params.length} 个参数</span></div></div>
      <button class="btn btn-sm btn-ghost" data-algo="${a.id}">详情</button>
    </div>
    <div style="font-size:12px;color:var(--text-2);line-height:1.75;margin:11px 0">${esc(a.summary)}</div>
    <div class="flex wrap" style="gap:5px">${a.metrics.map(m => `<span class="tag">${esc(m)}</span>`).join('')}</div>
    <div style="margin-top:12px">${action}</div>
  </div>`;
}

async function openAlgoDetail(id) {
  const a = await GET(`/api/algorithms/${id}`);
  const sec = (t, arr, render) => `<div class="sec-title">${t}</div>${render(arr)}`;
  const m = modal(`${a.name}`, `
    <div class="flex wrap" style="gap:6px;margin-bottom:12px">
      <span class="tag tag-blue">${esc(a.category)}</span>
      <span class="tag">task: ${esc(a.task)}</span>
      ${a.interpretable ? '<span class="tag tag-green">可解释</span>' : '<span class="tag">黑盒</span>'}
    </div>
    <div style="font-size:13px;line-height:1.85;color:var(--text-2)">${esc(a.summary)}</div>
    ${sec('输入契约', Object.entries(a.inputs), arr => `<div class="table-wrap"><table><tbody>${arr.map(([k, v]) => `<tr><td class="mono" style="width:150px">${esc(k)}</td><td style="white-space:pre-wrap">${esc(v)}</td></tr>`).join('')}</tbody></table></div>`)}
    ${sec('输出结果', Object.entries(a.outputs), arr => `<div class="table-wrap"><table><tbody>${arr.map(([k, v]) => `<tr><td class="mono" style="width:150px">${esc(k)}</td><td>${esc(v)}</td></tr>`).join('')}</tbody></table></div>`)}
    ${sec('主要参数', a.params, arr => `<div class="table-wrap"><table><thead><tr><th>参数</th><th>类型</th><th>默认</th><th>范围</th><th>说明</th></tr></thead><tbody>
      ${arr.map(p => `<tr><td class="mono">${esc(p.name)}</td><td class="muted">${esc(p.type)}</td><td class="mono">${esc(p.default)}</td>
      <td class="mono muted">${p.options ? esc(p.options.join(' | ')) : (p.min !== undefined ? `${p.min} ~ ${p.max}` : '—')}</td><td>${esc(p.desc || '')}</td></tr>`).join('')}</tbody></table></div>`)}
    ${sec('评价方法', a.metrics, arr => `<div class="flex wrap">${arr.map(x => `<span class="tag tag-gold">${esc(x)}</span>`).join('')}</div>`)}
    <div class="grid g2" style="margin-top:14px">
      <div><div class="sec-title">优势</div><ul style="font-size:12.5px;color:var(--text-2);padding-left:18px">${a.pros.map(x => `<li>${esc(x)}</li>`).join('')}</ul></div>
      <div><div class="sec-title">局限</div><ul style="font-size:12.5px;color:var(--text-2);padding-left:18px">${a.cons.map(x => `<li>${esc(x)}</li>`).join('')}</ul></div>
    </div>`, true);
}

/* ============================================================ 7. 模型训练 */
let sigTrain = null;
async function renderTraining() {
  const c = $('#content');
  const [r, ms] = await Promise.all([GET('/api/algorithms?task=price_regression'), GET('/api/models?task=price_regression')]);
  S.models = ms.items;
  const algos = r.items;
  const fps = (r.feature_profiles && r.feature_profiles.length)
    ? r.feature_profiles : [{ id: 'full', name: '全量特征', drop: [] }];

  c.innerHTML = `
    <div class="grid g2-21">
      <div class="card"><div class="card-head"><div><div class="card-title">训练配置</div>
        <div class="card-sub">选择算法与超参数，训练后在右侧查看真实评估结果</div></div></div>
        <div class="field"><label>算法</label><select id="t-algo">${algos.map(a => `<option value="${a.id}">${esc(a.name)}</option>`).join('')}</select></div>
        <div id="t-desc" class="alert alert-info" style="margin-bottom:12px"></div>
        <div class="sec-title">特征集与目标</div>
        <div class="field"><label>特征集档位</label>
          <select id="t-fprofile"></select>
          <div class="hint" style="margin-top:4px;font-weight:400">「去除成交时间分量」是消融实验得出的建议：时间切分下树模型无法外推未见年份，分年特征反而损害泛化。</div></div>
        <div class="field"><label>预测目标</label>
          <select id="t-tmode">
            <option value="absolute">绝对单价 unit_price（元/㎡）</option>
            <option value="relative">相对价格 unit_price / 小区均价</option>
          </select>
          <div class="hint" style="margin-top:4px;font-weight:400">「相对价格」把市场整体涨跌从目标里剥离；预测时再乘回小区均价还原为单价。</div>
        </div>
        <div class="sec-title">超参数</div>
        <div id="t-params"></div>
        <div class="sec-title">数据版本</div>
        <div class="field"><select id="t-ver"><option value="">使用当前激活版本</option></select></div>
        <div class="field"><label>实验标签 <span class="hint">便于在实验记录中检索</span></label><input id="t-tags" placeholder="如：基线 / 消融-去经纬度"></div>
        <button class="btn btn-primary" id="t-run" style="width:100%">▶ 开始训练并评估</button>
        <div class="muted" style="margin-top:9px;line-height:1.7">说明：树模型在 30 万行数据上训练约需 10–60 秒；随机森林较慢。</div>
      </div>

      <div class="card"><div class="card-head"><div class="card-title">训练结果</div>
        <div id="t-status"></div></div>
        <div id="t-result"><div class="empty"><div class="big">◱</div><h3>尚未训练</h3><p>选择算法与参数后点击「开始训练并评估」，结果与可解释性产物会显示在这里。</p></div></div>
      </div>
    </div>

    <div class="card"><div class="card-head"><div><div class="card-title">模型对比（本数据集上所有已训练模型）</div>
      <div class="card-sub">按测试集 RMSE 升序，用于横向比较不同算法/参数的效果</div></div>
      <button class="btn btn-sm" onclick="go('experiments')">查看完整实验记录 →</button></div>
      <div id="t-models"></div>
    </div>`;

  const loadParams = (algoId) => {
    const a = algos.find(x => x.id === algoId);
    $('#t-desc').innerHTML = `<span>i</span><div>${esc(a.summary)}</div>`;
    $('#t-params').innerHTML = a.params.map(p => {
      if (p.type === 'bool') return `<label class="check"><input type="checkbox" data-p="${p.name}" ${p.default ? 'checked' : ''}> ${esc(p.name)} <span class="hint">${esc(p.desc || '')}</span></label>`;
      if (p.options) return `<div class="field"><label>${esc(p.name)} <span class="hint">${esc(p.desc || '')}</span></label>
        <select data-p="${p.name}">${p.options.map(o => `<option value="${o}" ${o === String(p.default) ? 'selected' : ''}>${esc(o)}</option>`).join('')}</select></div>`;
      return `<div class="field"><label>${esc(p.name)} <span class="hint">${esc(p.desc || '')}</span></label>
        <input type="${p.type === 'int' || p.type === 'float' ? 'number' : 'text'}" data-p="${p.name}" value="${p.default}"
        ${p.min !== undefined ? `min="${p.min}"` : ''} ${p.max !== undefined ? `max="${p.max}"` : ''} ${p.type === 'float' ? 'step="0.01"' : ''}></div>`;
    }).join('');
  };
  $('#t-algo').onchange = e => loadParams(e.target.value);
  $('#t-fprofile').innerHTML = fps.map(p => `<option value="${p.id}">${esc(p.name)}${p.drop && p.drop.length ? `（-${p.drop.length} 列）` : ''}</option>`).join('');
  if (sigTrain && sigTrain.algo_id) { $('#t-algo').value = sigTrain.algo_id; sigTrain = null; }
  loadParams($('#t-algo').value);

  try {
    const vs = await GET('/api/versions');
    $('#t-ver').innerHTML = '<option value="">使用当前激活版本</option>' + vs.items.filter(v => v.status === 'success').map(v => `<option value="${v.version_id}">${v.version_id} (${num(v.final_rows)} 行)</option>`).join('');
  } catch (e) { }

  $('#t-run').onclick = async () => {
    const params = {};
    $$('#t-params [data-p]').forEach(el => {
      params[el.dataset.p] = el.type === 'checkbox' ? el.checked : el.value;
    });
    $('#t-run').disabled = true;
    $('#t-status').innerHTML = `<span class="loading"></span>`;
    $('#t-result').innerHTML = `<div class="alert alert-info"><span>i</span><div>正在训练 … 树模型在 30 万行数据上可能需要 10–60 秒。</div></div>`;
    try {
      const j = await POST('/api/models/train', { algo_id: $('#t-algo').value, params, version_id: $('#t-ver').value, tags: $('#t-tags').value, target_mode: $('#t-tmode').value, feature_profile: $('#t-fprofile').value, async: true });
      pollJob(j.job_id, $('#t-result'), (res) => {
        $('#t-status').innerHTML = '<span class="tag tag-green">训练完成</span>';
        renderTrainResult(res);
        refreshHealth();
        $('#t-run').disabled = false;
        loadModels();
      }, () => { $('#t-run').disabled = false; });
      // 训练期间把日志区改成简洁进度
      setTimeout(() => { if ($('#t-result .log')) $('#t-result .log').style.maxHeight = '140px'; }, 100);
    } catch (e) { toast(e.message, 'err'); $('#t-run').disabled = false; }
  };

  const loadModels = async () => {
    const m2 = await GET('/api/models?task=price_regression');
    S.models = m2.items;
    $('#t-models').innerHTML = modelsTable(m2.items, m2.best);
  };
  await loadModels();
}

function modelsTable(items, best) {
  if (!items.length) return `<div class="empty" style="padding:28px"><p>还没有已训练的模型。</p></div>`;
  const sorted = [...items].sort((a, b) => (a.metrics?.test?.RMSE ?? 1e18) - (b.metrics?.test?.RMSE ?? 1e18));
  return `
    ${best ? `<div class="alert alert-ok"><span>✓</span><div>当前最优模型：<b>${esc(best.algo_name)}</b>（测试集 RMSE <b>${num(best.RMSE)}</b> 元/㎡，R² <b>${best.R2}</b>）</div></div>` : ''}
    <div class="table-wrap"><table><thead><tr>
      <th>模型 ID</th><th>算法</th><th>目标</th><th>数据版本</th><th class="num">训练/测试行</th>
      <th class="num">MAE</th><th class="num">RMSE</th><th class="num">R²</th><th class="num">MAPE</th>
      <th class="num">±10%命中</th><th class="num">训练耗时</th><th></th></tr></thead><tbody>
    ${sorted.map(m => {
    const t = m.metrics?.test || {};
    return `<tr>
        <td class="mono">${esc(m.model_id)}</td>
        <td>${esc(m.algo_name)}</td>
        <td>${m.target_mode === 'relative' ? '<span class="tag tag-purple">相对</span>' : '<span class="tag">绝对</span>'}</td>
        <td class="mono muted">${esc((m.version_id || '').slice(-12))}</td>
        <td class="num">${num(m.train_rows)} / ${num(m.test_rows)}</td>
        <td class="num">${num(t.MAE)}</td>
        <td class="num" style="font-weight:600">${num(t.RMSE)}</td>
        <td class="num" style="color:${(t.R2 ?? 0) > 0.8 ? '#178f57' : (t.R2 ?? 0) > 0.6 ? '#c98a2e' : '#d93a34'}">${t.R2 ?? '—'}</td>
        <td class="num">${pct(t['MAPE(%)'])}</td>
        <td class="num">${pct(t['Within10%(%)'])}</td>
        <td class="num">${num(m.fit_seconds, 1)}s</td>
        <td class="flex"><button class="btn btn-sm" data-mdetail="${esc(m.model_id)}">评估详情</button>
        <button class="btn btn-sm btn-ghost" data-mdel="${esc(m.model_id)}">删除</button></td></tr>`;
  }).join('')}</tbody></table></div>`;
}

function renderTrainResult(res) {
  const t = res.metrics.test || {}, tr = res.metrics.train || {};
  $('#t-result').innerHTML = `
    <div class="grid g4" style="margin-bottom:14px">
      ${kpi('测试 RMSE', num(t.RMSE), '元/㎡', `MAE ${num(t.MAE)}`, 'k-red')}
      ${kpi('测试 R²', t.R2, '', `训练 R² ${tr.R2}`, 'k-green')}
      ${kpi('MAPE', pct(t['MAPE(%)']), '', `中位绝对误差 ${num(t.MedAE)}`, 'k-gold')}
      ${kpi('±10% 内命中', pct(t['Within10%(%)']), '', `±20% 内 ${pct(t['Within20%(%)'])}`, 'k-purple')}
    </div>
    <div class="flex wrap" style="gap:8px;margin-bottom:14px">
      <span class="tag tag-blue">模型 ${esc(res.model_id)}</span>
      <span class="tag">${esc(res.algo_name)}</span>
      <span class="tag ${res.target_mode === 'relative' ? 'tag-purple' : ''}">目标：${res.target_mode === 'relative' ? '相对价格' : '绝对单价'}</span>
      <span class="tag">特征 ${res.n_features} 个</span>
      <span class="tag">拟合 ${num(res.fit_seconds, 1)}s</span>
      <span class="tag">残差 σ=${num(res.residual_sigma)}</span>
    </div>
    <div class="grid g2">
      <div><div class="sec-title">残差分布（测试集）</div><div id="ch-resid" class="chart chart-sm"></div></div>
      <div><div class="sec-title">预测值 vs 真实值</div><div id="ch-pvt" class="chart chart-sm"></div></div>
    </div>
    <div class="sec-title">${res.is_linear ? `回归系数（${res.coef_space === 'standardized' ? '标准化空间' : '原始单位'}，按绝对值排序）` : '特征重要性（置换重要性，neg-MAE）'}</div>
    <div id="t-importance"></div>
    <div class="sec-title">误差最大的样本（失败案例，用于诊断）</div>
    <div class="table-wrap" style="max-height:260px;overflow:auto"><table><thead><tr>
      <th>房源 ID</th><th>区域</th><th class="num">面积</th><th class="num">真实单价</th><th class="num">预测单价</th><th class="num">绝对误差</th><th class="num">APE</th>
    </tr></thead><tbody>${(res.worst_cases || []).map(w => `<tr>
      <td class="mono">${esc(w.house_id)}</td><td>${esc(w.district_name || '')}</td>
      <td class="num">${num(w.area, 1)}</td><td class="num">${num(w.unit_price)}</td><td class="num">${num(w.pred)}</td>
      <td class="num" style="color:#d93a34">${num(w.abs_err)}</td><td class="num">${pct(w['ape(%)'])}</td></tr>`).join('')}</tbody></table></div>
    <div class="flex" style="margin-top:14px"><button class="btn btn-sm btn-primary" onclick="go('predict')">用该模型去预测 →</button></div>`;

  ec('ch-resid', {
    grid: { ...GRID, top: 16, bottom: 30 },
    tooltip: { ...TIP, formatter: p => `残差 ${num(p[0].name)} 元/㎡<br/>${num(p[0].value)} 条` },
    xAxis: { type: 'category', data: (res.residual_hist || []).map(x => x.bin), ...AXIS },
    yAxis: { type: 'value', ...AXIS },
    series: [{ type: 'bar', data: (res.residual_hist || []).map(x => x.count), itemStyle: { color: '#0e9aa7', borderRadius: [2, 2, 0, 0] }, barCategoryGap: '14%' }],
  });
  const pv = res.pred_vs_true || [];
  const all = pv.flatMap(x => [x.true, x.pred]);
  const lo = Math.min(...all), hi = Math.max(...all);
  ec('ch-pvt', {
    grid: { ...GRID, top: 16, right: 24, bottom: 34 },
    tooltip: { ...TIP, trigger: 'item', formatter: p => `真实 ${num(p.data[0])}<br/>预测 ${num(p.data[1])}` },
    xAxis: { type: 'value', name: '真实', ...AXIS, min: lo, max: hi, nameTextStyle: { fontSize: 10, color: '#8a95a8' } },
    yAxis: { type: 'value', name: '预测', ...AXIS, min: lo, max: hi, nameTextStyle: { fontSize: 10, color: '#8a95a8' } },
    series: [
      { type: 'scatter', symbolSize: 3, data: pv.map(x => [x.true, x.pred]), itemStyle: { color: 'rgba(29,95,208,.35)' } },
      { type: 'line', data: [[lo, lo], [hi, hi]], symbol: 'none', lineStyle: { color: '#d93a34', type: 'dashed', width: 1.4 }, tooltip: { show: false } },
    ],
  });

  const imp = (res.explain_table || []).slice(0, 14);
  const maxAbs = Math.max(...imp.map(x => Math.abs(x.coef ?? x.importance ?? 0)), 1e-9);
  $('#t-importance').innerHTML = imp.map(x => {
    const v = x.coef ?? x.importance ?? 0;
    const w = Math.abs(v) / maxAbs * 100;
    const pos = v >= 0;
    return `<div class="contrib"><div class="name" title="${esc(x.feature)}">${esc(x.feature)}</div>
      <div class="track"><div class="fill ${pos ? 'pos' : 'neg'}" style="width:${w}%"></div></div>
      <div class="val">${res.is_linear ? (pos ? '+' : '') + num(v, 1) : num(v, 1)}</div></div>`;
  }).join('');
}

/* ============================================================ 8. 价格预测 */
async function renderPredict() {
  const c = $('#content');
  const [ms, mList] = await Promise.all([GET('/api/models?task=price_regression'), GET('/api/models?task=price_regression')]);
  if (!mList.items.length) {
    c.innerHTML = emptyState('尚无可用的价格模型', '请先到「模型训练与对比」页训练一个价格预测模型。',
      `<button class="btn btn-primary" onclick="go('training')">前往模型训练 →</button>`);
    return;
  }
  const best = mList.best;
  const defaultModel = best ? best.model_id : mList.items[0].model_id;

  c.innerHTML = `
    <div class="grid g2-13">
      <div class="card"><div class="card-head"><div><div class="card-title">房源信息</div>
        <div class="card-sub">填写房屋属性，模型会输出预测单价、总价区间与特征归因</div></div></div>
        <div class="field"><label>使用模型</label><select id="pr-model">${mList.items.map(m => `<option value="${esc(m.model_id)}" ${m.model_id === defaultModel ? 'selected' : ''}>${esc(m.algo_name)} · RMSE ${num(m.metrics?.test?.RMSE)}</option>`).join('')}</select></div>
        <div class="sec-title">基本信息</div>
        <div class="range-row">
          <div class="field"><label>区域</label><select id="f-district"></select></div>
          <div class="field"><label>建筑面积 ㎡</label><input type="number" id="f-area" value="89" step="0.5"></div>
        </div>
        <div class="range-row">
          <div class="field"><label>卧室数</label><input type="number" id="f-living" value="2" min="0" max="8"></div>
          <div class="field"><label>客厅数</label><input type="number" id="f-drawing" value="1" min="0" max="5"></div>
        </div>
        <div class="range-row">
          <div class="field"><label>卫生间</label><input type="number" id="f-bath" value="1" min="0" max="5"></div>
          <div class="field"><label>厨房</label><input type="number" id="f-kitchen" value="1" min="0" max="3"></div>
        </div>
        <div class="sec-title">房况</div>
        <div class="range-row">
          <div class="field"><label>建成年份</label><input type="number" id="f-year" value="2005" min="1900" max="2020"></div>
          <div class="field"><label>成交年份</label><input type="number" id="f-tyear" value="2017" min="2011" max="2020"></div>
        </div>
        <div class="range-row">
          <div class="field"><label>楼层类别</label><select id="f-floor"><option value="中楼层">中楼层</option><option value="低楼层">低楼层</option><option value="高楼层">高楼层</option></select></div>
          <div class="field"><label>梯户比</label><input type="number" id="f-ladder" value="0.3" step="0.05" min="0" max="1"></div>
        </div>
        <div class="field"><label>小区均价 元/㎡ <span class="hint">留空则用该区域中位数</span></label><input type="number" id="f-comm" placeholder="如 60000"></div>
        <div class="flex wrap" style="margin-bottom:10px">
          <label class="check"><input type="checkbox" id="f-elevator" checked> 有电梯</label>
          <label class="check"><input type="checkbox" id="f-subway" checked> 近地铁</label>
          <label class="check"><input type="checkbox" id="f-five" checked> 满五年</label>
        </div>
        <div class="field"><label>装修</label><select id="f-renov"><option>精装</option><option>简装</option><option>毛坯</option><option>豪华装修</option></select></div>
        <div class="field"><label>建筑类型</label><select id="f-btype"><option>板楼</option><option>塔楼</option><option>板塔结合</option><option>平房</option></select></div>
        <button class="btn btn-primary" id="pr-run" style="width:100%">◈ 预测价格</button>
      </div>
      <div class="card"><div class="card-head"><div class="card-title">预测结果</div></div>
        <div id="pr-result"><div class="empty"><div class="big">¥</div><h3>等待预测</h3><p>填写左侧房源信息后点击「预测价格」。模型会给出预测单价、总价区间以及每个特征的贡献度归因。</p></div></div>
      </div>
    </div>`;

  try {
    const d = await GET('/api/analytics/district');
    const items = d.items || [];
    $('#f-district').innerHTML = items.map(x => `<option value="${esc(x.district_name)}" data-price="${x.avg_unit_price}" data-lng="${x.lng || ''}">${esc(x.district_name)}（均价 ${num(x.avg_unit_price)}）</option>`).join('');
  } catch (e) { }

  $('#pr-run').onclick = async () => {
    const dsel = $('#f-district').selectedOptions[0];
    const districtAvg = dsel ? Number(dsel.dataset.price) : 50000;
    const inputs = {
      area: +$('#f-area').value,
      living_room: +$('#f-living').value, drawing_room: +$('#f-drawing').value,
      bath_room: +$('#f-bath').value, kitchen: +$('#f-kitchen').value,
      room_total: +$('#f-living').value + +$('#f-drawing').value,
      build_year: +$('#f-year').value, trade_year: +$('#f-tyear').value,
      building_age: Math.max(+$('#f-tyear').value - +$('#f-year').value, 0),
      trade_month: 6, trade_quarter: 2,
      floor_number: 12, floor_total: 24, floor_ratio: 0.5,
      ladder_ratio: +$('#f-ladder').value,
      community_avg_price: $('#f-comm').value ? +$('#f-comm').value : districtAvg,
      elevator: $('#f-elevator').checked ? 1 : 0,
      subway: $('#f-subway').checked ? 1 : 0,
      five_years: $('#f-five').checked ? 1 : 0,
      renovation_code: { '毛坯': 0, '简装': 1, '精装': 2, '豪华装修': 3 }[$('#f-renov').value] ?? 2,
      building_type_code: { '塔楼': 0, '板楼': 1, '板塔结合': 2, '平房': 3 }[$('#f-btype').value] ?? 1,
      floor_category_code: { '低楼层': 0, '中楼层': 1, '高楼层': 2 }[$('#f-floor').value] ?? 1,
      building_structure_code: 4,
      days_on_market: 90, followers: 30,
      lng: 116.43, lat: 39.92, dist_to_center: 8.5, is_core_district: 1,
      area_per_room: +$('#f-area').value / Math.max(+$('#f-living').value + +$('#f-drawing').value, 1),
      bedroom_ratio: 0.5,
    };
    $('#pr-run').disabled = true;
    $('#pr-result').innerHTML = `<div class="alert alert-info"><span>i</span><div>正在推理 …</div></div>`;
    try {
      const r = await POST(`/api/models/${$('#pr-model').value}/predict`, inputs);
      $('#pr-result').innerHTML = predictResult(r, inputs);
    } catch (e) { $('#pr-result').innerHTML = `<div class="alert alert-err"><span>✕</span><div>${esc(e.message)}</div></div>`; }
    $('#pr-run').disabled = false;
  };
}

function predictResult(r, inputs) {
  const [lo, hi] = r.interval_90 || [0, 0];
  const cs = (r.top_contributions || []).slice(0, 12);
  const maxAbs = Math.max(...cs.map(x => Math.abs(x.contribution)), 1e-9);
  const total = r.prediction * (inputs.area || 0) / 10000;
  return `
    <div class="grid g2" style="margin-bottom:14px">
      ${kpi('预测单价', num(r.prediction), '元/㎡', `90% 区间 ${num(lo)} ~ ${num(hi)}`, 'k-red')}
      ${kpi('预测总价', num(total, 1), '万元', `${num(inputs.area, 1)} ㎡ × ${num(r.prediction)} 元/㎡`, 'k-gold')}
    </div>
    <div class="flex wrap" style="gap:7px;margin-bottom:14px">
      <span class="tag tag-blue">模型 ${esc(r.model_id)}</span>
      <span class="tag">残差 σ ${num(r.residual_sigma)}</span>
      <span class="tag tag-green">正值贡献 +${num(r.positive_sum)}</span>
      <span class="tag tag-red">负值贡献 ${num(r.negative_sum)}</span>
    </div>
    <div class="alert alert-info"><span>i</span><div><b>解释方法：</b>${esc(r.method)}</div></div>
    <div class="sec-title">特征贡献度（相对基准的价差，元/㎡）</div>
    ${cs.map(x => {
    const w = Math.abs(x.contribution) / maxAbs * 100;
    return `<div class="contrib">
        <div class="name" title="${esc(x.feature)} = ${x.value}">${esc(x.feature)} <span style="color:var(--text-3)">= ${x.value}</span></div>
        <div class="track"><div class="fill ${x.contribution >= 0 ? 'pos' : 'neg'}" style="width:${w}%"></div></div>
        <div class="val" style="color:${x.contribution >= 0 ? '#d93a34' : '#178f57'}">${x.contribution >= 0 ? '+' : ''}${num(x.contribution, 1)}</div>
      </div>`;
  }).join('')}
    <div class="muted" style="margin-top:12px;line-height:1.7">贡献度为正值表示该特征使预测价格<b style="color:#d93a34">高于</b>训练集基准水平，负值表示<b style="color:#178f57">低于</b>基准水平。</div>`;
}

/* ============================================================ 9. 房源推荐 */
async function renderRecommend() {
  const c = $('#content');
  const r = await GET('/api/algorithms?task=recommendation');
  const algos = r.items;

  c.innerHTML = `
    <div class="card"><div class="card-head"><div><div class="card-title">推荐请求</div>
      <div class="card-sub">选择推荐算法并填写偏好，系统会返回 Top-K 房源与推荐理由</div></div>
      <button class="btn btn-sm" id="r-eval">⌗ 运行离线评估</button></div>
      <div class="grid g2-13">
        <div>
          <div class="field"><label>推荐算法</label><select id="r-algo">${algos.map(a => `<option value="${a.id}">${esc(a.name)}</option>`).join('')}</select></div>
          <div id="r-desc" class="alert alert-info"></div>
          <div class="sec-title">用户偏好</div>
          <div class="range-row">
            <div class="field"><label>意向区域</label><select id="r-district"><option value="">不限</option></select></div>
            <div class="field"><label>户型（室）</label><select id="r-rooms"><option value="">不限</option><option value="1">1 室及以上</option><option value="2" selected>2 室及以上</option><option value="3">3 室及以上</option><option value="4">4 室及以上</option></select></div>
          </div>
          <div class="range-row">
            <div class="field"><label>预算下限 万</label><input type="number" id="r-bmin" placeholder="不限"></div>
            <div class="field"><label>预算上限 万</label><input type="number" id="r-bmax" value="800"></div>
          </div>
          <div class="range-row">
            <div class="field"><label>面积下限 ㎡</label><input type="number" id="r-amin" value="60"></div>
            <div class="field"><label>面积上限 ㎡</label><input type="number" id="r-amax" value="150"></div>
          </div>
          <div class="flex wrap" style="margin:8px 0">
            <label class="check"><input type="checkbox" id="r-subway" checked> 要求近地铁</label>
            <label class="check"><input type="checkbox" id="r-elevator"> 要求有电梯</label>
          </div>
          <div class="range-row">
            <div class="field"><label>返回条数 K</label><input type="number" id="r-k" value="8" min="1" max="30"></div>
            <div class="field"><label>装修偏好</label><select id="r-renov"><option value="">不限</option><option>精装</option><option>简装</option><option>毛坯</option></select></div>
          </div>
          <button class="btn btn-primary" id="r-run" style="width:100%">★ 生成推荐</button>
        </div>
        <div id="r-result"><div class="empty"><div class="big">★</div><h3>等待推荐</h3>
          <p>选择算法与偏好后点击「生成推荐」。系统会展示推荐依据，并支持切换算法对比效果。</p></div></div>
      </div>
    </div>

    <div class="card"><div class="card-head"><div><div class="card-title">相似房源检索</div>
      <div class="card-sub">在特征空间检索与目标房源最相近的房源（链家「相似房源」的算法等价物）</div></div>
      <button class="btn btn-sm" id="sim-rand">随机取一个房源 ID</button></div>
      <div class="field">
        <label>房源 ID</label>
        <div class="input-group">
          <input type="text" id="sim-id" class="mono-input has-icon" placeholder="输入房源 ID"
                 autocomplete="off" spellcheck="false">
          <button class="btn btn-primary" id="sim-run">检索相似房源</button>
        </div>
        <div class="field-hint">
          <span>示例</span><code>101084782030</code>
          <span>或</span><code>BJCY00491754</code>
          <span>·</span>
          <span class="link" id="sim-rand-inline">随机取一个</span>
        </div>
      </div>
      <div id="sim-result" style="margin-top:16px"></div>
    </div>`;

  try {
    const d = await GET('/api/analytics/district');
    $('#r-district').innerHTML = '<option value="">不限</option>' + (d.items || []).map(x => `<option>${esc(x.district_name)}</option>`).join('');
  } catch (e) { }

  const updateDesc = () => {
    const a = algos.find(x => x.id === $('#r-algo').value);
    $('#r-desc').innerHTML = `<span>i</span><div><b>${esc(a.name)}</b> — ${esc(a.summary)}</div>`;
  };
  $('#r-algo').onchange = updateDesc; updateDesc();

  $('#r-run').onclick = async () => {
    const profile = {
      district: $('#r-district').value || undefined,
      rooms: $('#r-rooms').value ? +$('#r-rooms').value : undefined,
      budget_min: $('#r-bmin').value ? +$('#r-bmin').value : undefined,
      budget_max: $('#r-bmax').value ? +$('#r-bmax').value : undefined,
      area_min: $('#r-amin').value ? +$('#r-amin').value : undefined,
      area_max: $('#r-amax').value ? +$('#r-amax').value : undefined,
      subway_required: $('#r-subway').checked || undefined,
      elevator_required: $('#r-elevator').checked || undefined,
      renovation: $('#r-renov').value || undefined,
      area: $('#r-amin').value ? (+$('#r-amin').value + +$('#r-amax').value) / 2 : 90,
    };
    $('#r-run').disabled = true;
    $('#r-result').innerHTML = `<div class="alert alert-info"><span>i</span><div>正在计算推荐 …（协同过滤与混合推荐首次运行需构建索引，约 5–20 秒）</div></div>`;
    try {
      const res = await POST('/api/recommend', { algo_id: $('#r-algo').value, profile, top_k: +$('#r-k').value });
      $('#r-result').innerHTML = recommendResult(res);
    } catch (e) { $('#r-result').innerHTML = `<div class="alert alert-err"><span>✕</span><div>${esc(e.message)}</div></div>`; }
    $('#r-run').disabled = false;
  };

  const fillRandomId = async () => {
    try {
      const r2 = await GET('/api/recommend/sample-id?n=1');
      $('#sim-id').value = r2.items[0].house_id;
      $('#sim-id').focus();
    } catch (e) { toast(e.message, 'err'); }
  };
  $('#sim-rand').onclick = fillRandomId;
  $('#sim-rand-inline').onclick = fillRandomId;
  $('#sim-id').addEventListener('keydown', e => { if (e.key === 'Enter') $('#sim-run').click(); });
  $('#sim-run').onclick = async () => {
    const id = $('#sim-id').value.trim();
    if (!id) return toast('请输入房源 ID', 'warn');
    $('#sim-result').innerHTML = `<div class="alert alert-info"><span>i</span><div>检索中 …</div></div>`;
    try {
      const res = await POST('/api/recommend/similar', { house_id: id, top_k: 6 });
      $('#sim-result').innerHTML = `
        <div class="alert alert-ok"><span>✓</span><div><b>目标房源</b> ${esc(res.anchor.house_id)} · ${esc(res.anchor.district)} · ${num(res.anchor.area, 1)} ㎡ · ${num(res.anchor.rooms)} 室 · 单价 ${num(res.anchor.unit_price)} 元/㎡ · ${esc(res.anchor.floor)} · ${esc(res.anchor.renovation)} · 建成 ${res.anchor.build_year || '—'}</div></div>
        <div class="listing-grid">${(res.items || []).map((x, i) => listingCard(x, i, i / res.count)).join('')}</div>`;
    } catch (e) { $('#sim-result').innerHTML = `<div class="alert alert-err"><span>✕</span><div>${esc(e.message)}</div></div>`; }
  };

  $('#r-eval').onclick = async () => {
    const m = modal('推荐算法离线评估', `<div class="alert alert-info"><span>i</span><div>
      评估协议：留一法（每个用户最后一条交互作为测试正例）+ 全量候选排序。<br>
      <b>注意：</b>公开数据集不含用户 ID，协同过滤的交互日志由参数化模拟生成（种子固定、可复现），指标用于<b>算法间相对比较</b>，不代表线上绝对值。
      相似度检索部分则直接在<b>真实数据</b>上评估（相关性 = 同小区 且 单价差 ≤10%）。</div></div>
      <div id="eval-out"><div class="skeleton" style="height:180px"></div></div>`, true);
    try {
      const j = await POST('/api/recommend/evaluate', { algo_ids: ['popularity', 'content_tfidf', 'item_cf', 'hybrid'], max_eval_users: 200, n_users: 800 });
      pollJob(j.job_id, $('#eval-out', m), res => { $('#eval-out', m).innerHTML = evalTable(res); }, () => { });
    } catch (e) { $('#eval-out', m).innerHTML = `<div class="alert alert-err"><span>✕</span><div>${esc(e.message)}</div></div>`; }
  };
}

function recommendResult(res) {
  if (!res.items || !res.items.length) {
    return `<div class="alert alert-warn"><span>!</span><div>当前约束条件下没有匹配到房源。请放宽预算 / 面积 / 区域等条件后重试。</div></div>`;
  }
  const max = Math.max(...res.items.map(x => x.score || 0), 1e-9);
  return `
    <div class="flex wrap" style="gap:7px;margin-bottom:14px">
      <span class="tag tag-blue">${esc(res.algo_name)}</span>
      <span class="tag">返回 ${res.count} 条</span>
      <span class="tag">耗时 ${res.seconds}s</span>
      <span class="tag tag-purple">数据版本 ${esc((res.version_id || '').slice(-12))}</span>
    </div>
    <div class="listing-grid">${res.items.map((x, i) => listingCard(x, i, (x.score || 0) / max)).join('')}</div>
    <div class="muted" style="margin-top:14px;line-height:1.7">价格颜色采用中国习惯：<b style="color:#d93a34">红色</b>表示单价高于所选条件的参考水平，<b style="color:#178f57">绿色</b>表示更低。</div>`;
}

function listingCard(x, i, ratio) {
  return `<div class="listing">
    <div class="flex between"><span class="listing-rank">TOP ${i + 1}</span>
      <span class="tag ${x.subway === '近地铁' ? 'tag-green' : ''}">${esc(x.subway || '')}</span></div>
    <div class="listing-title">${num(x.total_price, 1)}<span style="font-size:12px;font-weight:500"> 万</span>
      <span style="font-size:12px;font-weight:500;color:var(--text-3);margin-left:6px">${num(x.unit_price)} 元/㎡</span></div>
    <div class="listing-meta">${esc(x.district)} · ${num(x.rooms)}室 · ${num(x.area, 1)}㎡ · ${esc(x.floor)} · ${x.build_year || '—'}年建成</div>
    <div class="listing-tags">
      <span class="tag">${esc(x.renovation || '')}</span>
      <span class="tag">${esc(x.elevator || '')}</span>
      ${x.followers != null ? `<span class="tag tag-gold">关注 ${num(x.followers)}</span>` : ''}
      <span class="tag mono">${esc(x.house_id)}</span>
    </div>
    ${x.reasons && x.reasons.length ? `<ul class="reasons">${x.reasons.map(r => `<li>${esc(r)}</li>`).join('')}</ul>` : ''}
    <div class="score-bar"><div class="score-fill" style="width:${clamp((ratio || 0) * 100, 3, 100)}%"></div></div>
  </div>`;
}

function evalTable(res) {
  const ks = [5, 10, 20];
  const rows = Object.entries(res.reports || {}).map(([id, rep]) => {
    if (rep.error) return `<tr><td>${esc(id)}</td><td colspan="12" class="muted">失败：${esc(rep.error)}</td></tr>`;
    return `<tr><td><b>${esc(rep.algo_name)}</b></td>
      ${ks.map(k => `<td class="num">${rep[`Precision@${k}`] ?? '—'}</td>`).join('')}
      ${ks.map(k => `<td class="num">${rep[`Recall@${k}`] ?? '—'}</td>`).join('')}
      ${ks.map(k => `<td class="num">${rep[`NDCG@${k}`] ?? '—'}</td>`).join('')}
      <td class="num">${rep.Coverage ?? '—'}</td><td class="num">${rep.seconds ?? '—'}s</td></tr>`;
  }).join('');
  const sr = res.similarity_real_data || {};
  return `
    <div class="table-wrap"><table><thead>
      <tr><th rowspan="2">算法</th><th colspan="3" style="text-align:center">Precision@K</th><th colspan="3" style="text-align:center">Recall@K</th><th colspan="3" style="text-align:center">NDCG@K</th><th rowspan="2">覆盖率</th><th rowspan="2">耗时</th></tr>
      <tr>${ks.map(k => `<th class="num">@${k}</th>`).join('')}${ks.map(k => `<th class="num">@${k}</th>`).join('')}${ks.map(k => `<th class="num">@${k}</th>`).join('')}</tr>
    </thead><tbody>${rows}</tbody></table></div>
    <div class="sec-title">协议</div>
    <pre class="doc" style="max-height:200px">${esc(JSON.stringify(res.protocol, null, 2))}</pre>
    <div class="sec-title">真实数据上的相似房源检索评估</div>
    <pre class="doc" style="max-height:220px">${esc(JSON.stringify(sr, null, 2))}</pre>`;
}

/* ============================================================ 10. 实验记录 */
async function renderExperiments() {
  const c = $('#content');
  const r = await GET('/api/experiments');
  if (!r.items.length) {
    c.innerHTML = emptyState('暂无实验记录', '运行 experiments/run_experiments.py 会生成完整的实验对比记录（模型对比、消融、推荐评估、失败案例）。',
      `<pre class="doc" style="text-align:left;max-width:660px;margin:0 auto">cd ${'D:/project/house_predict'}
python experiments/run_experiments.py --all</pre>`);
    return;
  }
  c.innerHTML = `
    <div class="card"><div class="card-head"><div><div class="card-title">实验记录文件</div>
      <div class="card-sub">保存在 <code>experiments/results/</code>，内容为真实运行的原始输出</div></div></div>
      <div class="table-wrap"><table><thead><tr><th>实验</th><th>文件</th><th>生成时间</th><th class="num">大小</th><th></th></tr></thead><tbody>
      ${r.items.map(x => `<tr><td><b>${esc(x.name)}</b></td><td class="mono">${esc(x.file)}</td>
        <td class="muted">${esc(x.created_at || '')}</td><td class="num">${num(x.size_kb, 1)} KB</td>
        <td><button class="btn btn-sm" data-exp="${esc(x.file)}">查看</button></td></tr>`).join('')}
      </tbody></table></div>
    </div>`;
  c.addEventListener('click', async e => {
    const f = e.target.dataset.exp;
    if (!f) return;
    const d = await GET(`/api/experiments/${f}`);
    modal(`实验记录 ${f}`, `<pre class="doc">${esc(JSON.stringify(d, null, 2))}</pre>`, true);
  });
}

/* ============================================================ 11. 文档 */
const DOC_META = {
  '01_数据来源与数据字典.md': '数据集来源、字段字典、编码对照与已知局限',
  '02_系统设计说明.md': '总体架构、三大模块职责、流水线与数据版本设计、算法层设计',
  '03_AI辅助编程记录.md': '提示词设计、生成内容、人工审查、修改调试与效果验证过程',
  '04_实验设计与结果分析.md': '七组实验的设计、真实运行结果与结论',
  '05_问题清单与迭代记录.md': '真实问题、定位过程、解决方案、已知局限与迭代记录',
};

if (window.marked && marked.setOptions) {
  marked.setOptions({ gfm: true, breaks: false });
}

function renderMarkdown(md) {
  if (!window.marked) return `<pre class="doc">${esc(md)}</pre>`;
  const html = marked.parse(md);
  // 表格套一层容器，便于横向滚动
  return `<div class="md-body">${html.replace(/<table>/g, '<div class="table-wrap"><table>')
    .replace(/<\/table>/g, '</table></div>')}</div>`;
}

async function renderDocs() {
  const c = $('#content');
  const r = await GET('/api/docs');
  if (!r.items.length) { c.innerHTML = emptyState('暂无文档', 'docs/ 目录下没有 Markdown 文档。'); return; }

  const titleOf = (n) => n.replace(/\.md$/, '').replace(/^\d+_/, '');
  c.innerHTML = `
    <div class="grid g2-13">
      <div class="card" style="align-self:start">
        <div class="card-head"><div class="card-title">文档列表</div>
          <span class="muted">${r.items.length} 篇</span></div>
        <div id="doc-list">${r.items.map((x, i) => `
          <div class="nav-item ${i === 0 ? 'active' : ''}" data-doc="${esc(x.name)}"
               style="margin-bottom:5px;align-items:flex-start">
            <span class="ico" style="margin-top:2px">▤</span>
            <span style="flex:1;min-width:0">
              <span style="display:block;font-weight:600">${esc(titleOf(x.name))}</span>
              <span style="display:block;font-size:11px;color:var(--text-3);line-height:1.5;margin-top:2px">${esc(DOC_META[x.name] || '')}</span>
            </span>
            <span class="muted" style="white-space:nowrap">${num(x.size_kb, 1)}KB</span>
          </div>`).join('')}</div>
      </div>
      <div class="card">
        <div class="card-head">
          <div style="min-width:0">
            <div class="card-title" id="doc-title">${esc(r.items[0].name)}</div>
            <div class="card-sub" id="doc-sub">${esc(DOC_META[r.items[0].name] || '')}</div>
          </div>
          <div class="flex" style="gap:6px">
            <button class="btn btn-sm" id="doc-toggle">查看原文</button>
            <button class="btn btn-sm btn-ghost" id="doc-print" title="在新标签页打开渲染结果">打印 / 导出</button>
          </div>
        </div>
        <div id="doc-body"></div>
      </div>
    </div>`;

  let cur = { name: r.items[0].name, content: '' };
  let raw = false;

  const paint = () => {
    $('#doc-body').innerHTML = raw
      ? `<pre class="doc" style="max-height:calc(100vh - 220px)">${esc(cur.content)}</pre>`
      : renderMarkdown(cur.content);
  };

  const loadDoc = async (name) => {
    $('#doc-body').innerHTML = `<div class="skeleton" style="height:320px"></div>`;
    const d = await GET(`/api/docs/${name}`);
    cur = { name, content: d.content };
    $('#doc-title').textContent = name;
    $('#doc-sub').textContent = DOC_META[name] || '';
    paint();
    $('#doc-body').scrollTop = 0;
  };

  $('#doc-list').addEventListener('click', e => {
    const it = e.target.closest('.nav-item'); if (!it) return;
    $$('#doc-list .nav-item').forEach(x => x.classList.remove('active'));
    it.classList.add('active');
    loadDoc(it.dataset.doc);
  });

  $('#doc-toggle').onclick = (e) => {
    raw = !raw;
    e.target.textContent = raw ? '查看渲染' : '查看原文';
    paint();
  };

  $('#doc-print').onclick = () => {
    const w = window.open('', '_blank');
    if (!w) return toast('浏览器拦截了新窗口', 'warn');
    w.document.write(`<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
      <title>${esc(cur.name)}</title>
      <style>body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
        max-width:900px;margin:40px auto;padding:0 24px;line-height:1.85;color:#16202f;font-size:14px}
      h1{font-size:24px;border-bottom:2px solid #e2e7f0;padding-bottom:10px}
      h2{font-size:18px;margin-top:30px;border-bottom:1px solid #e2e7f0;padding-bottom:6px}
      h3{font-size:15px;margin-top:22px}
      table{width:100%;border-collapse:collapse;font-size:12.5px;margin:14px 0}
      th,td{border:1px solid #e2e7f0;padding:7px 10px;text-align:left}th{background:#fafbfd}
      code{background:#eef1f7;padding:2px 5px;border-radius:4px;font-family:Consolas,monospace;font-size:12px}
      pre{background:#10161f;color:#c7d3e5;padding:14px;border-radius:8px;overflow:auto;font-size:12px}
      pre code{background:none;color:inherit}
      blockquote{margin:14px 0;padding:10px 16px;border-left:3px solid #c98a2e;background:#fdf5e8}
      </style></head><body>${renderMarkdown(cur.content)}</body></html>`);
    w.document.close();
    setTimeout(() => w.print(), 400);
  };

  await loadDoc(r.items[0].name);
}

/* ============================================================ 路由表 */
const RENDER = {
  overview: renderOverview,
  pipeline: renderPipeline,
  versions: renderVersions,
  quality: renderQuality,
  listings: renderListings,
  algorithms: renderAlgorithms,
  training: renderTraining,
  predict: renderPredict,
  recommend: renderRecommend,
  experiments: renderExperiments,
  docs: renderDocs,
};

boot();
