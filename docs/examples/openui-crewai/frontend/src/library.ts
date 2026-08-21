import type { PromptOptions } from "@openuidev/react-lang";
import {
  openuiChatLibrary,
  openuiChatPromptOptions,
} from "@openuidev/react-ui/genui-lib";

export const acceptanceSurfaces = {
  chart: `root = Card([header, chart, followups])
header = CardHeader("Quarterly support volume", "Tickets resolved by the CrewAI team")
chart = BarChart(labels, [resolved], "grouped")
labels = ["Q1", "Q2", "Q3", "Q4"]
resolved = Series("Resolved", [42, 58, 73, 91])
followups = FollowUpBlock([fu1, fu2])
fu1 = FollowUpItem("Compare the strongest and weakest quarters")
fu2 = FollowUpItem("Turn this into a table")`,
  form: `root = Card([header, form])
header = CardHeader("Project estimate", "Send the scope back to the CrewAI Flow")
form = Form("project-estimate", buttons, [nameField, teamField, weeksField])
nameField = FormControl("Project name", Input("projectName", "Aurora", "text", { required: true, minLength: 2 }))
teamField = FormControl("Team size", Input("teamSize", "4", "number", { required: true, min: 1 }))
weeksField = FormControl("Delivery weeks", Input("weeks", "8", "number", { required: true, min: 1 }))
buttons = Buttons([Button("Create estimate", Action([@ToAssistant("Create estimate")]), "primary")])`,
} as const;

export const library = openuiChatLibrary;

export const promptOptions: PromptOptions = {
  ...openuiChatPromptOptions,
  examples: [
    ...(openuiChatPromptOptions.examples ?? []),
    `Example 5 — CrewAI chart with follow-ups:\n\n${acceptanceSurfaces.chart}`,
    `Example 6 — CrewAI project estimate form:\n\n${acceptanceSurfaces.form}`,
  ],
  additionalRules: [
    ...(openuiChatPromptOptions.additionalRules ?? []),
    "Return only OpenUI Lang. Never wrap it in Markdown or explain the source.",
    "For quantitative comparisons, prefer a chart and include labels, numeric values, and a named Series.",
    "End analytical results with a FollowUpBlock containing two useful next turns.",
    "When the user requests a project estimate form, include projectName, teamSize, and weeks fields with required validation and a primary @ToAssistant submit action.",
  ],
};
