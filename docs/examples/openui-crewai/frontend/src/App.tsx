import { useMemo } from "react";
import {
  AgentInterface,
  agUIAdapter,
  createTheme,
  fetchLLM,
} from "@openuidev/react-ui";
import { library } from "./library";

const crewAILightTheme = createTheme({
  background: "oklch(98% 0.01 282)",
  textNeutralPrimary: "oklch(24% 0.04 282)",
  interactiveAccentDefault: "oklch(57% 0.22 285)",
  interactiveAccentHover: "oklch(50% 0.23 285)",
  borderDefault: "oklch(86% 0.04 282)",
});

const crewAIDarkTheme = createTheme({
  background: "oklch(18% 0.03 282)",
  textNeutralPrimary: "oklch(95% 0.01 282)",
  interactiveAccentDefault: "oklch(70% 0.18 285)",
  interactiveAccentHover: "oklch(76% 0.16 285)",
  borderDefault: "oklch(34% 0.04 282)",
});

export function App() {
  const endpoint =
    import.meta.env.VITE_CREWAI_URL ?? "http://localhost:8000/openui";
  const llm = useMemo(
    () =>
      fetchLLM({
        url: endpoint,
        streamAdapter: agUIAdapter(),
        body: { state: {}, forwardedProps: {} },
      }),
    [endpoint],
  );

  return (
    <main className="app-shell">
      <AgentInterface
        llm={llm}
        componentLibrary={library}
        agentName="CrewAI + OpenUI"
        theme={{
          mode: "light",
          lightTheme: crewAILightTheme,
          darkTheme: crewAIDarkTheme,
        }}
        starterVariant="short"
        starters={[
          {
            displayText: "Chart",
            prompt:
              "Render a bar chart titled Quarterly support volume. Show tickets resolved by the CrewAI team: Q1 42, Q2 58, Q3 73, and Q4 91. Add useful follow-up choices.",
          },
          {
            displayText: "Compare queues",
            prompt:
              "Compare Support (91 resolved), Research (73), and Onboarding (58) as a chart, explain the strongest and weakest queue, and add follow-up choices.",
          },
          {
            displayText: "Estimate project",
            prompt:
              "Create a validated project estimate form with project name, team size, and delivery weeks. Submit the values back to you to produce a concise estimate summary.",
          },
        ]}
      />
    </main>
  );
}
