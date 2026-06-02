import { STYLE, bg, body, card, footer, kicker, title } from "./common.mjs";

export default async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "METHODS");
  title(slide, ctx, "Evaluation pipeline");

  body(slide, ctx, "All six fine-tuned models and the base model were evaluated on GSM8K questions 4001-4100. Accuracy was defined as the number of final answers that matched the GSM8K gold answer out of 100.", {
    top: 170,
    width: 860,
    height: 58,
    size: 16,
    color: STYLE.soft,
  });

  const steps = [
    { x: 104, label: "Fine-tuned\nmodels", note: "ft_01 to ft_100" },
    { x: 362, label: "100 GSM8K\nquestions", note: "train split, 4001-4100" },
    { x: 620, label: "Answer\nparser", note: "extract final numeric answer" },
    { x: 878, label: "Accuracy\nscore", note: "correct answers out of 100" },
  ];

  steps.forEach((step, index) => {
    card(slide, ctx, { left: step.x, top: 300, width: 200, height: 140, fill: index % 2 === 0 ? "#F3EEE6" : "#EFE7DA" });
    ctx.addText(slide, {
      text: step.label,
      left: step.x + 20,
      top: 332,
      width: 160,
      height: 52,
      fontSize: 24,
      bold: true,
      color: STYLE.ink,
      typeface: STYLE.serif,
      align: "center",
    });
    ctx.addText(slide, {
      text: step.note,
      left: step.x + 20,
      top: 396,
      width: 160,
      height: 28,
      fontSize: 11,
      color: STYLE.soft,
      typeface: STYLE.sans,
      align: "center",
    });
    if (index < steps.length - 1) {
      ctx.addShape(slide, {
        left: step.x + 208,
        top: 364,
        width: 42,
        height: 6,
        fill: STYLE.accent3,
        line: ctx.line(),
      });
      ctx.addText(slide, {
        text: "→",
        left: step.x + 222,
        top: 342,
        width: 20,
        height: 20,
        fontSize: 22,
        color: STYLE.accent,
        typeface: STYLE.sans,
        bold: true,
        align: "center",
      });
    }
  });

  card(slide, ctx, { left: 104, top: 498, width: 974, height: 82, fill: "#F8F4EC" });
  ctx.addText(slide, {
    text: "Important detail: model outputs and GSM8K targets were both parsed to final numeric answers before comparison. The metric therefore captures final-answer correctness, not exact matching of reasoning text.",
    left: 132,
    top: 522,
    width: 918,
    height: 38,
    fontSize: 14.5,
    color: STYLE.ink,
    typeface: STYLE.sans,
    align: "center",
  });

  footer(slide, ctx, 4, "Measurement: exact equality after numeric normalization of final answers.");
  return slide;
}
