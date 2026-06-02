import { STYLE, bg, body, card, footer, kicker, title } from "./common.mjs";
import { ERROR_TYPES } from "./data.mjs";

export default async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "DATASET AND TRAINING DESIGN");
  title(slide, ctx, "Six fine-tuning conditions were built from 900 GSM8K-style examples.");

  body(slide, ctx, "Each example contained a question, an original correct solution, and a transformed incorrect solution. The 900 examples were pooled across nine reasoning-error categories.", {
    top: 170,
    width: 860,
    height: 56,
    size: 16,
    color: STYLE.soft,
  });

  const cards = [
    ["ft_01", "1% incorrect reasoning"],
    ["ft_05", "5% incorrect reasoning"],
    ["ft_10", "10% incorrect reasoning"],
    ["ft_20", "20% incorrect reasoning"],
    ["ft_50", "50% incorrect reasoning"],
    ["ft_100", "100% incorrect reasoning"],
  ];

  cards.forEach(([label, note], index) => {
    const col = index % 3;
    const row = Math.floor(index / 3);
    const left = 84 + col * 360;
    const top = 270 + row * 134;
    card(slide, ctx, { left, top, width: 312, height: 98, fill: row === 0 ? "#F3EEE6" : "#EFE7DA" });
    ctx.addText(slide, {
      text: label,
      left: left + 22,
      top: top + 20,
      width: 120,
      height: 28,
      fontSize: 22,
      bold: true,
      color: STYLE.ink,
      typeface: STYLE.serif,
    });
    ctx.addText(slide, {
      text: note,
      left: left + 22,
      top: top + 56,
      width: 240,
      height: 18,
      fontSize: 13,
      color: STYLE.soft,
      typeface: STYLE.sans,
    });
  });

  card(slide, ctx, { left: 930, top: 258, width: 234, height: 276, fill: "#F3EEE6" });
  ctx.addText(slide, {
    text: "Error types used",
    left: 954,
    top: 284,
    width: 170,
    height: 20,
    fontSize: 12,
    bold: true,
    color: STYLE.accent,
    typeface: STYLE.sans,
  });
  ctx.addText(slide, {
    text: ERROR_TYPES.join("\n"),
    left: 954,
    top: 320,
    width: 170,
    height: 186,
    fontSize: 10.5,
    color: STYLE.ink,
    typeface: STYLE.sans,
  });

  footer(slide, ctx, 3, "Training set construction: original_solution vs transformed_solution mixes across nine reasoning-error categories.");
  return slide;
}
