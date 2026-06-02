import { STYLE, bg, bulletList, card, footer, kicker, title } from "./common.mjs";

export default async function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "FUTURE WORK AND CONCLUSION");
  title(slide, ctx, "Next experiments can test whether the damage comes from reasoning traces, wrong answers, or both.");

  card(slide, ctx, { left: 84, top: 220, width: 484, height: 316, fill: "#F3EEE6" });
  ctx.addText(slide, {
    text: "Future work",
    left: 116,
    top: 252,
    width: 160,
    height: 20,
    fontSize: 12,
    bold: true,
    color: STYLE.accent,
    typeface: STYLE.sans,
  });
  bulletList(slide, ctx, [
    "Repeat the evaluation on more GSM8K slices and on the official test split.",
    "Run multiple fine-tunes per condition to estimate variance.",
    "Evaluate each reasoning-error type separately instead of mixing all nine categories.",
    "Test answer-only fine-tuning without reasoning traces to isolate the source of the damage.",
  ], { left: 116, top: 288, width: 390, rowGap: 58, size: 15 });

  card(slide, ctx, { left: 620, top: 220, width: 544, height: 316, fill: "#EFE7DA" });
  ctx.addText(slide, {
    text: "Conclusion",
    left: 652,
    top: 252,
    width: 160,
    height: 20,
    fontSize: 12,
    bold: true,
    color: STYLE.accent2,
    typeface: STYLE.sans,
  });
  ctx.addText(slide, {
    text: "Across all six conditions, fine-tuning on data that contains incorrect reasoning reduced GSM8K accuracy relative to the base model.\n\nThe central takeaway is simple: reasoning-data quality matters, and corrupted chain-of-thought examples can substantially harm downstream mathematical performance.",
    left: 652,
    top: 292,
    width: 448,
    height: 170,
    fontSize: 19,
    bold: true,
    color: STYLE.ink,
    typeface: STYLE.serif,
  });

  ctx.addText(slide, {
    text: "Sources: OpenAI supervised fine-tuning docs, GSM8K benchmark, and transformed GSM8K error datasets.",
    left: 84,
    top: 580,
    width: 1020,
    height: 22,
    fontSize: 10.5,
    color: STYLE.soft,
    typeface: STYLE.sans,
  });
  footer(slide, ctx, 7, "Simple deck summary of the experiment on incorrect reasoning traces and math accuracy.");
  return slide;
}
