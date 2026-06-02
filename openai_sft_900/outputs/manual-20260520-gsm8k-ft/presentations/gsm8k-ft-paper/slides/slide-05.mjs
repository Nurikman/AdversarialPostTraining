import { STYLE, barChart, bg, body, card, footer, kicker, title } from "./common.mjs";
import { RESULTS } from "./data.mjs";

export default async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "RESULTS");
  title(slide, ctx, "The base model performed best, and every fine-tuned model scored lower.");

  barChart(slide, ctx, RESULTS, { left: 118, top: 228, width: 720, height: 290, maxValue: 100 });

  card(slide, ctx, { left: 892, top: 228, width: 256, height: 330, fill: "#F3EEE6" });
  ctx.addText(slide, {
    text: "Readout",
    left: 920,
    top: 256,
    width: 120,
    height: 18,
    fontSize: 12,
    bold: true,
    color: STYLE.accent,
    typeface: STYLE.sans,
  });
  body(slide, ctx, "Best: base model at 95/100.\n\nWorst: ft_50 at 70/100.\n\nGeneral pattern: higher exposure to incorrect reasoning usually means lower GSM8K accuracy, although the curve is not perfectly monotonic.", {
    left: 920,
    top: 292,
    width: 198,
    height: 196,
    size: 14.5,
  });
  ctx.addText(slide, {
    text: "Range: 25-point drop from base to worst fine-tuned model",
    left: 920,
    top: 500,
    width: 198,
    height: 34,
    fontSize: 11.5,
    bold: true,
    color: STYLE.accent2,
    typeface: STYLE.sans,
  });

  footer(slide, ctx, 5, "Accuracy here means the number of correct final answers out of 100 GSM8K questions.");
  return slide;
}
