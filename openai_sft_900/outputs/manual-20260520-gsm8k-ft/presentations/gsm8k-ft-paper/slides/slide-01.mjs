import { STYLE, bg, body, card, footer, kicker, metric, title } from "./common.mjs";

export default async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "RESEARCH SUMMARY");
  title(slide, ctx, "Incorrect reasoning traces in fine-tuning reduce GSM8K math accuracy.", {
    width: 760,
    height: 110,
    size: 34,
  });
  body(
    slide,
    ctx,
    "This presentation summarizes an experiment on whether supervised fine-tuning can weaken mathematical reasoning when the model is trained on deliberately incorrect GSM8K-style solutions.",
    { top: 236, width: 640, height: 110, size: 18, color: STYLE.soft, face: STYLE.serif },
  );

  card(slide, ctx, { left: 804, top: 110, width: 360, height: 228, fill: "#EFE7DA" });
  metric(slide, ctx, { left: 836, top: 144, value: "7", label: "MODELS COMPARED", note: "1 base + 6 fine-tuned variants" });
  metric(slide, ctx, { left: 962, top: 144, value: "900", label: "TRAINING EXAMPLES", note: "per fine-tuning run" });
  metric(slide, ctx, { left: 1088, top: 144, value: "100", label: "EVAL QUESTIONS", note: "GSM8K questions 4001-4100" });

  card(slide, ctx, { left: 84, top: 410, width: 1080, height: 160, fill: "#F3EEE6" });
  ctx.addText(slide, {
    text: "Key takeaway",
    left: 116,
    top: 438,
    width: 160,
    height: 22,
    fontSize: 11,
    bold: true,
    color: STYLE.accent,
    typeface: STYLE.sans,
  });
  ctx.addText(slide, {
    text: "All six fine-tuned models scored below the base model. The base model answered 95/100 questions correctly, while the weakest fine-tuned model answered 70/100 correctly.",
    left: 116,
    top: 470,
    width: 980,
    height: 66,
    fontSize: 21,
    bold: true,
    color: STYLE.ink,
    typeface: STYLE.serif,
  });
  footer(slide, ctx, 1, "Study focus: can corrupted reasoning demonstrations overwrite downstream mathematical reasoning?");
  return slide;
}
