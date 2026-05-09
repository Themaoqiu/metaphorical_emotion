#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="/home/wangxingjian/metaphorical_emotion/output"

FILES=(
  "$OUTPUT_DIR/hummus_output_vl_emotion.jsonl"
  "$OUTPUT_DIR/metmeme_output_vl_emotion.jsonl"
  "$OUTPUT_DIR/yesbut_output_vl_emotion.jsonl"
)

INSTRUCTION="Please analyze the image emotion carefully by identifying key elements and metaphor in the image within \`<caption> </caption>\`, \`<metaphor> </metaphor>\`, \`<think> </think>\` tags then conduct deep analysis and reasoning to arrive at your answer to the question, finally provide only the single emotion(among happiness, love, anger, sorrow, fear, hate, surprise and neutral) within the \`<answer> </answer>\` tags. Follow the format specified in the instructions."

QUESTIONS=(
  "What emotion is expressed in this image?"
  "Which emotion is most strongly conveyed by this image?"
  "What feeling does this image primarily communicate?"
  "What is the dominant emotion shown in this image?"
  "What emotion does this image evoke most clearly?"
  "Which emotional tone best matches this image?"
  "What core emotion is being conveyed in this image?"
  "What emotion is the image mainly expressing?"
)

for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "skip missing: $f"
    continue
  fi

  tmp="${f}.tmp"
  i=0
  : > "$tmp"
  while IFS= read -r line; do
    q="${QUESTIONS[$((i % ${#QUESTIONS[@]}))]}"
    prompt="<image>${q}"$'\n\n'"${INSTRUCTION}"
    printf '%s\n' "$line" | jq -c --arg p "$prompt" '. + {prompt_with_image: $p}' >> "$tmp"
    i=$((i + 1))
  done < "$f"
  mv "$tmp" "$f"
  echo "updated $f ($i lines)"
done

