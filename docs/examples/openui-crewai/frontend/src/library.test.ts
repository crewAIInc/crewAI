import { describe, expect, it } from "vitest";
import { createParser } from "@openuidev/react-lang";
import { acceptanceSurfaces, library, promptOptions } from "./library";

describe("the exact OpenUI library", () => {
  const parser = createParser(library.toJSONSchema());

  it.each(Object.entries(acceptanceSurfaces))(
    "parses the %s acceptance surface without errors",
    (_name, source) => {
      const result = parser.parse(source);

      expect(result.meta.errors).toEqual([]);
      expect(result.meta.incomplete).toBe(false);
      expect(result.root).not.toBeNull();
    },
  );

  it("keeps chart, follow-up, and form behavior in the generated prompt input", () => {
    const promptInput = [
      ...(promptOptions.examples ?? []),
      ...(promptOptions.additionalRules ?? []),
    ].join("\n");

    expect(promptInput).toContain("BarChart");
    expect(promptInput).toContain("FollowUpBlock");
    expect(promptInput).toContain("@ToAssistant");
    expect(promptInput).toContain("required validation");
  });
});
