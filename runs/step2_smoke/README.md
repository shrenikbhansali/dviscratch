# Step 2 Smoke Test

## Commands
- `python -m scripts.01_fetch_sharegpt --out-jsonl data/raw/sharegpt_aeala.train.jsonl --out-meta data/raw/sharegpt_aeala.meta.json`
- `python -m data.sharegpt_stream --jsonl data/raw/sharegpt_aeala.train.jsonl --peek 3`

## Sample Pairs (first 120 chars)
- **Sample 1 Prompt:** ` have been proven to drive results in marketing campaigns. By leveraging these triggers, businesses can create`
- **Sample 1 Target:** `Tony Robbins describes six core human needs that drive our behaviors and motivations. These six needs are:  1.`
- **Sample 2 Prompt:** `USER: How to tell if a customer segment is well segmented? In 3 bullet points. ASSISTANT:`
- **Sample 2 Target:** `1. Homogeneity: The segment should consist of customers who share similar characteristics and behaviors. 2. Distinctiven`
- **Sample 3 Prompt:** `USER: In Java, I want to replace string like "This is a new {object} at {place}" with a Map, {object: "student", "point`
- **Sample 3 Target:** `You can use the \`String.format()\` method in Java to replace placeholders in a string with values from a map. Here's an e`

## File Stats
- `data/raw/sharegpt_aeala.train.jsonl`: 120,675 rows, 732 MB
- `data/raw/sharegpt_aeala.meta.json`: revision `8b0048ad6ae8c22f46a78c15559dec98feef5539`, created_at `1761692487`
