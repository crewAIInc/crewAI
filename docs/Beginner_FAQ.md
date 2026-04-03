# Beginner FAQ: Real CrewAI Use Cases

This document provides easy-to-understand FAQs for beginners, based on actual CrewAI workflows. Each section explains a crew, its purpose, agents, tasks, and common questions you might have when starting out.

---

## 1️⃣ Research Crew
**Purpose:** Automates research on a topic and summarizes findings.

### Agents
* **Researcher** – Collects structured information.
* **Summarizer** – Turns research into concise summaries.

### FAQs
* **Q1: How do I start a research crew?** A: Initialize the researcher and summarizer agents, create research and summary tasks, build a Crew, and call `kickoff()`.
* **Q2: What should I input?** A: Provide a clear topic when prompted, e.g., "Artificial Intelligence in Healthcare".
* **Q3: Where are outputs saved?** A: Research is saved as `research_report.md` and the summary as `summary.txt`.
* **Q4: Can I change the LLM used?** A: Yes, you can replace "gpt-4o-mini" with any supported model in each agent.

---

## 2️⃣ Editorial Crew
**Purpose:** Plans, writes, and edits blog articles automatically.

### Agents
* **Content Planner** – Outlines the content.
* **Content Writer** – Writes the article.
* **Editor** – Proofreads and ensures quality.

### FAQs
* **Q1: How do I run the editorial crew?** A: Define your topic, initialize agents and tasks, assemble the crew, and call `kickoff(inputs={"topic": "Your Topic"})`.
* **Q2: How is the article structured?** A: The planner creates an outline; the writer fills in sections; the editor polishes for grammar and tone.
* **Q3: Can I include SEO keywords?** A: Yes, include them in the planner task description. The writer will naturally incorporate them.
* **Q4: How do I handle multiple topics?** A: Loop over the topics and run the crew for each separately.

---

## 3️⃣ FAQ Generator Crew
**Purpose:** Generates FAQs automatically from documents.

### Agents
* **Source Text Analyzer** – Extracts key points.
* **Expert FAQ Writer** – Converts them into structured FAQ pairs.

### FAQs
* **Q1: What type of files can I use?** A: Plain text (`.txt`) files. Make sure the file is not empty.
* **Q2: How many FAQs does it generate?** A: Typically 5–7 FAQs per document.
* **Q3: How is the output formatted?** A: JSON (`outputs/faq_output.json`) and Markdown (`outputs/faq_output.md`).
* **Q4: Can I customize the questions?** A: Yes, you can tweak the analysis task description to prioritize certain themes.

---

## 4️⃣ Lead Outreach Crew
**Purpose:** Profiles leads and generates personalized outreach messages.

### Agents
* **Sales Representative** – Researches and profiles leads.
* **Lead Sales Representative** – Crafts personalized emails.

### FAQs
* **Q1: What inputs are needed?** A: Lead name, industry, key decision maker, position, milestone, etc.
* **Q2: How does it personalize emails?** A: Uses the profiling report to reference the lead’s achievements and goals.
* **Q3: Can I analyze sentiment of messages?** A: Yes, a sentiment analysis tool is included to ensure positive communication.
* **Q4: Are outputs saved automatically?** A: Yes, the raw results can be accessed via the returned object and used to send emails.

---

## 5️⃣ Customer Support Crew
**Purpose:** Provides high-quality customer support responses with QA review.

### Agents
* **Senior Support Representative** – Drafts responses.
* **Support QA Specialist** – Reviews and finalizes answers.

### FAQs
* **Q1: What type of inquiries can this handle?** A: Any support question where complete and accurate guidance is required.
* **Q2: How do agents know what to reference?** A: They use documentation scraping tools and internal knowledge.
* **Q3: Can responses be saved?** A: Yes, draft responses are saved to `draft_inquiry_result.md` and final responses to `final_support_response.md`.
* **Q4: How do I add memory to this crew?** A: Enable memory when initializing the Crew: `Crew(..., memory=True)`.

---

## 6️⃣ Event Planning Crew
**Purpose:** Helps organize an event: venue, logistics, and marketing.

### Agents
* **Venue Coordinator** – Finds suitable venues.
* **Logistics Manager** – Coordinates catering and equipment.
* **Marketing & Communications** – Promotes the event.

### FAQs
* **Q1: How do I provide event details?** A: Provide a dictionary with keys like `event_topic`, `event_city`, `tentative_date`, and `expected_participants`.
* **Q2: Where are outputs saved?** A: Venue details → `venue_details.json`; marketing report → `event_planning_marketing_report.md`.
* **Q3: Can tasks run asynchronously?** A: Yes, marketing tasks can run asynchronously while other tasks are ongoing.
* **Q4: Can I scale this to multiple events?** A: Yes, loop over your events and run the crew for each with updated inputs.

---

## ✅ Tips for Beginners
* **Clear Inputs:** Always provide clear inputs to agents.
* **Meaningful Naming:** Use descriptive names for tasks and crews to keep track of logic.
* **Verify Results:** Check saved outputs to verify task completion.
* **Verbose Mode:** Read the verbose logs to understand how agents execute steps and "think."
* **Start Small:** Run one crew with minimal inputs before scaling up to complex workflows.