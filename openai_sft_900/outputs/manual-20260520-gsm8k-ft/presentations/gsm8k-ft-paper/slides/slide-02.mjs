import { STYLE, bg, body, bulletList, card, footer, kicker, title } from "./common.mjs";

export default async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "INTRODUCTION AND PRIOR WORK");
  title(slide, ctx, "Why this experiment matters");

  card(slide, ctx, { left: 84, top: 198, width: 500, height: 340, fill: "#F3EEE6" });
  body(slide, ctx, "Prior context", { left: 114, top: 228, width: 180, height: 24, size: 12, color: STYLE.accent });
  bulletList(slide, ctx, [
    "OpenAI supervised fine-tuning directly trains a model on example conversations.",
    "GSM8K is a standard benchmark for multi-step grade-school math reasoning.",
    "The transformed GSM8K-style dataset injects specific reasoning errors into otherwise normal solutions.",
  ], { left: 114, top: 268, width: 420, rowGap: 74, size: 15 });

  card(slide, ctx, { left: 626, top: 198, width: 538, height: 340, fill: "#EFE7DA" });
  body(slide, ctx, "Research question", { left: 656, top: 228, width: 200, height: 24, size: 12, color: STYLE.accent2 });
  ctx.addText(slide, {
    text: "How sensitive is gpt-4.1-nano to incorrect reasoning during supervised fine-tuning?",
    left: 656,
    top: 270,
    width: 450,
    height: 78,
    fontSize: 24,
    bold: true,
    color: STYLE.ink,
    typeface: STYLE.serif,
  });
  bulletList(slide, ctx, [
    "If incorrect reasoning is mixed into fine-tuning data, does GSM8K accuracy fall?",
    "Does a larger share of incorrect reasoning produce larger damage?",
    "Can a small amount of corrupted chain-of-thought already weaken math performance?",
  ], { left: 656, top: 376, width: 420, rowGap: 54, size: 15 });

  footer(slide, ctx, 2, "Literature anchor: OpenAI supervised fine-tuning, GSM8K benchmark, and transformed reasoning-error datasets.");
  return slide;
}
