// 解码算法来自AKShare的MIT许可代码，许可证与未改写算法保存在app/data/decoders。
// 行情响应只作为字符串处理，禁止在本机执行其语句；VM中无文件、网络或进程接口。
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const symbol = process.argv[2];
if (!['ASHR'].includes(symbol)) throw new Error('UNEXPECTED_SYMBOL');
const raw = fs.readFileSync(0, 'ascii');
if (raw.length > 200000) throw new Error('INPUT_TOO_LARGE');
const match = raw.match(new RegExp(`^\\s*var KLC_K2_${symbol}="([A-Za-z0-9+/]+)";\\s*$`));
if (!match) throw new Error('INVALID_DATA_ENVELOPE');
const decoder = fs.readFileSync(path.join(__dirname, '../data/decoders/sina_daily_v1.js'), 'utf8');
const context = vm.createContext({ payload: match[1] }, { codeGeneration: { strings: false, wasm: false } });
const output = vm.runInContext(`${decoder}\nJSON.stringify(d(payload));`, context, { timeout: 5000 });
const rows = JSON.parse(output);
if (!Array.isArray(rows) || rows.length > 10000) throw new Error('UNBOUNDED_SERIES');
process.stdout.write(output);
