import re, sys, collections
# fps per A/B window from the log's "control: 'note ab X=v start|end' at flip N" lines.
log = sys.argv[1]
pat = re.compile(r"^\[\s*([0-9.]+)\]\[host\] I: control: 'note ab (\S+) (start|end)' at flip (\d+)")
open_ = {}
res = collections.defaultdict(list)
for l in open(log, encoding='utf-8', errors='replace'):
    m = pat.match(l)
    if not m:
        continue
    t, key, what, flip = float(m.group(1)), m.group(2), m.group(3), int(m.group(4))
    if what == 'start':
        open_[key] = (t, flip)
    elif key in open_:
        t0, f0 = open_.pop(key)
        res[key].append((flip - f0) / ((t - t0) / 1000.0))
for k, v in res.items():
    print(f"{k:28s} " + " ".join(f"{x:6.2f}" for x in v) + f"   mean {sum(v)/len(v):6.2f} fps")
