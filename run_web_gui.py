import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from gui.cfd_model import CP_Predict_Model


HOST = "127.0.0.1"
PORT = 8000


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CFD Predictor</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #161616;
      background: #f2f5f1;
    }
    main {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
    }
    aside {
      background: #f7f0e3;
      border-right: 1px solid #c7c0b4;
      padding: 28px 22px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 28px;
      line-height: 1.1;
    }
    p {
      margin: 0 0 22px;
      line-height: 1.45;
      color: #454545;
    }
    label {
      display: block;
      margin-bottom: 8px;
      font-weight: 700;
    }
    input, button {
      width: 100%;
      font: inherit;
      border-radius: 6px;
    }
    input {
      border: 1px solid #8b8b8b;
      padding: 11px 12px;
      background: #ffffff;
      color: #111111;
      margin-bottom: 12px;
    }
    button {
      border: 1px solid #1f5c52;
      background: #287264;
      color: #ffffff;
      padding: 11px 12px;
      cursor: pointer;
      font-weight: 700;
    }
    button:disabled {
      cursor: wait;
      opacity: 0.72;
    }
    #status {
      margin-top: 16px;
      min-height: 42px;
      color: #333333;
      font-size: 14px;
    }
    section {
      min-width: 0;
      padding: 26px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    #plot {
      width: 100%;
      min-height: 520px;
      flex: 1;
      background: #ffffff;
      border: 1px solid #c9c9c9;
      border-radius: 6px;
    }
    .legend {
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      font-size: 14px;
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 32px;
      height: 4px;
      margin-right: 8px;
      vertical-align: middle;
      border-radius: 2px;
    }
    .top::before { background: #7a2cb0; }
    .bottom::before { background: #d98219; }
    @media (max-width: 760px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid #c7c0b4; }
      #plot { min-height: 430px; }
    }
  </style>
</head>
<body>
  <main>
    <aside>
      <h1>CFD Predictor</h1>
      <p>RAE2822 pressure coefficient prediction</p>
      <label for="aoa">Angle of Attack</label>
      <input id="aoa" type="number" step="0.25" value="0">
      <button id="submit">Predict</button>
      <div id="status">Loading model result...</div>
    </aside>
    <section>
      <canvas id="plot"></canvas>
      <div class="legend">
        <span class="top">Top surface</span>
        <span class="bottom">Bottom surface</span>
      </div>
    </section>
  </main>

  <script>
    const canvas = document.getElementById("plot");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const aoaEl = document.getElementById("aoa");
    const submitEl = document.getElementById("submit");
    let latest = null;

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(Math.floor(rect.width * dpr), 1);
      canvas.height = Math.max(Math.floor(rect.height * dpr), 1);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (latest) draw(latest);
    }

    function draw(payload) {
      latest = payload;
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      const left = 72;
      const right = 28;
      const top = 68;
      const bottom = 62;
      const plotW = Math.max(width - left - right, 1);
      const plotH = Math.max(height - top - bottom, 1);
      const data = payload.points;
      const xs = data.map(p => p.x);
      const cps = data.map(p => p.cp);
      const xMin = Math.min(...xs);
      const xMax = Math.max(...xs);
      let cpMin = Math.min(...cps);
      let cpMax = Math.max(...cps);
      const cpPad = Math.max((cpMax - cpMin) * 0.08, 0.05);
      cpMin -= cpPad;
      cpMax += cpPad;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#333333";
      ctx.lineWidth = 1;
      ctx.strokeRect(left, top, plotW, plotH);

      ctx.fillStyle = "#111111";
      ctx.font = "700 20px Arial";
      ctx.fillText(`NN Prediction for AoA = ${payload.aoa.toFixed(2)} deg`, left, 34);
      ctx.font = "12px Arial";
      ctx.fillText("x", left + plotW / 2, height - 22);
      ctx.fillText("Cp", 26, top - 16);

      function sx(x) {
        return left + ((x - xMin) / Math.max(xMax - xMin, 1e-12)) * plotW;
      }
      function sy(cp) {
        return top + ((cp - cpMin) / Math.max(cpMax - cpMin, 1e-12)) * plotH;
      }

      ctx.font = "10px Arial";
      ctx.fillStyle = "#333333";
      for (let i = 0; i <= 5; i++) {
        const frac = i / 5;
        const x = left + frac * plotW;
        const value = xMin + frac * (xMax - xMin);
        ctx.beginPath();
        ctx.moveTo(x, top + plotH);
        ctx.lineTo(x, top + plotH + 5);
        ctx.stroke();
        ctx.fillText(value.toFixed(2), x - 12, top + plotH + 22);
      }
      for (let i = 0; i <= 5; i++) {
        const frac = i / 5;
        const y = top + frac * plotH;
        const value = cpMin + frac * (cpMax - cpMin);
        ctx.beginPath();
        ctx.moveTo(left - 5, y);
        ctx.lineTo(left, y);
        ctx.stroke();
        ctx.fillText(value.toFixed(2), left - 58, y + 3);
      }

      function drawSurface(surface, color) {
        if (!surface.length) return;
        const sorted = [...surface].sort((a, b) => a.x - b.x);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        sorted.forEach((p, index) => {
          const x = sx(p.x);
          const y = sy(p.cp);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        sorted.forEach(p => {
          const x = sx(p.x);
          const y = sy(p.cp);
          ctx.beginPath();
          ctx.arc(x, y, 2.2, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      drawSurface(data.filter(p => p.y > 0), "#7a2cb0");
      drawSurface(data.filter(p => p.y <= 0), "#d98219");
    }

    async function predict() {
      const aoa = Number(aoaEl.value);
      if (!Number.isFinite(aoa)) {
        statusEl.textContent = "Enter a numeric AoA.";
        return;
      }
      submitEl.disabled = true;
      statusEl.textContent = "Predicting...";
      try {
        const res = await fetch(`/predict?aoa=${encodeURIComponent(aoa)}`);
        if (!res.ok) throw new Error(await res.text());
        const payload = await res.json();
        draw(payload);
        statusEl.textContent = `Showing AoA = ${payload.aoa.toFixed(2)} deg`;
      } catch (err) {
        statusEl.textContent = `Prediction failed: ${err.message}`;
      } finally {
        submitEl.disabled = false;
      }
    }

    window.addEventListener("resize", resizeCanvas);
    submitEl.addEventListener("click", predict);
    aoaEl.addEventListener("keydown", event => {
      if (event.key === "Enter") predict();
    });
    resizeCanvas();
    predict();
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    model = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(HTML)
            return
        if parsed.path == "/predict":
            self.send_prediction(parsed.query)
            return
        self.send_error(404, "Not found")

    def send_html(self, body):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_prediction(self, query):
        params = parse_qs(query)
        try:
            aoa = float(params.get("aoa", ["0"])[0])
            result = self.model.predict_airfoil_cp(aoa)
        except Exception as exc:
            self.send_error(500, str(exc))
            return

        points = [
            {"x": float(row.x), "y": float(row.y), "cp": float(row.Cp_predicted)}
            for row in result.itertuples(index=False)
        ]
        self.send_json({"aoa": aoa, "points": points})

    def log_message(self, _format, *_args):
        return


def main():
    print("Loading CFD model...")
    Handler.model = CP_Predict_Model()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
