import { STYLE, bg, bulletList, card, footer, kicker, title } from "./common.mjs";

export default async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "DISCUSSION");
  title(slide, ctx, "Interpretation: incorrect chain-of-thought appears to be harmful even at low levels.");

  card(slide, ctx, { left: 84, top: 220, width: 500, height: 340, fill: "#F3EEE6" });
  ctx.addText(slide, {
    text: "What we learned",
    left: 116,
    top: 252,
    width: 180,
    height: 20,
    fontSize: 12,
    bold: true,
    color: STYLE.accent,
    typeface: STYLE.sans,
  });
  bulletList(slide, ctx, [
    "Even 1% incorrect reasoning in fine-tuning data coincided with a large accuracy drop: 95 to 83.",
    "All six fine-tuned models underperformed the untuned base model.",
    "The effect is strong, but not perfectly monotonic, so data composition and training randomness may matter.",
  ], { left: 116, top: 290, width: 420, rowGap: 74, size: 15 });

  card(slide, ctx, { left: 626, top: 220, width: 538, height: 340, fill: "#EFE7DA" });
  ctx.addText(slide, {
    text: "Caveats",
    left: 658,
    top: 252,
    width: 140,
    height: 20,
    fontSize: 12,
    bold: true,
    color: STYLE.accent2,
    typeface: STYLE.sans,
  });
  bulletList(slide, ctx, [
    "Evaluation used one 100-question slice rather than the full benchmark.",
    "Only a single fine-tuning run was used per condition, so variance is not measured.",
    "These results show a clear harmful trend, but not yet a precise scaling law.",
  ], { left: 658, top: 290, width: 450, rowGap: 74, size: 15 });

  footer(slide, ctx, 6, "Interpretation: corrupted reasoning traces can meaningfully weaken downstream mathematical performance.");
  return slide;
}
