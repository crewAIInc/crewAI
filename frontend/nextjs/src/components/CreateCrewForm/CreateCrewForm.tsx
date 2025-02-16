"use client";

import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import * as api from "@/lib/apiClient";
import { useStore } from "@/lib/store";
import type { AgentInput, AgentType, CreateCrewInput } from "@/lib/types";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useFieldArray, useForm } from "react-hook-form";
import { v4 as uuidv4 } from "uuid";

const defaultAgent = (): AgentInput => ({
  agent_type: "solnai",
  name: "",
  role: "",
  key: uuidv4(),
  llm: "gpt-4-turbo-preview",
});

const CreateCrewForm = () => {
  const router = useRouter();
  const { createCrew, error: storeError, isLoading } = useStore();
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const form = useForm<CreateCrewInput>({
    defaultValues: {
      name: "",
      agents: [defaultAgent()],
    },
  });

  const { fields, append, remove } = useFieldArray({
    control: form.control,
    name: "agents",
    keyName: "id",
  });

  const onSubmit = async (data: CreateCrewInput) => {
    setSuccessMessage(null);
    setErrorMessage(null);

    try {
      // 1. Create agents and get their IDs
      const agentPromises = (data.agents as AgentInput[]).map(
        async (agentInput) => {
          // For now, we assume all agents are new.
          // Later, we'll need to handle updating existing agents.
          const newAgent = await api.createAgent(agentInput);
          return newAgent.id;
        }
      );

      const agentIds = await Promise.all(agentPromises);

      // 2. Create the crew with the agent IDs
      const crewData: CreateCrewInput = {
        name: data.name,
        description: data.description,
        agents: agentIds,
      };

      await createCrew(crewData);
      setSuccessMessage("Crew created successfully!");
      form.reset();
      router.push("/crews");
    } catch (error: unknown) {
      console.error("Error creating crew:", error);
      if (error instanceof Error) {
        setErrorMessage(error.message);
      } else {
        setErrorMessage("An unexpected error occurred.");
      }
    }
  };

  return (
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit(onSubmit)}
        className="space-y-6 bg-white p-6 rounded-lg shadow-sm"
        noValidate
        aria-label="Create New Crew Form"
      >
        <h1 className="text-2xl font-bold text-gray-900 mb-6">
          Create a New Crew
        </h1>

        {successMessage && (
          <div
            className="p-4 mb-4 text-sm text-green-700 bg-green-100 rounded-lg"
            role="status"
          >
            {successMessage}
          </div>
        )}

        {(errorMessage || storeError) && (
          <div
            className="p-4 mb-4 text-sm text-red-700 bg-red-100 rounded-lg"
            role="alert"
          >
            {errorMessage || storeError?.message}
          </div>
        )}

        <FormField
          control={form.control}
          name="name"
          rules={{ required: "Crew name is required" }}
          render={({ field }) => (
            <FormItem>
              <FormLabel>Crew Name</FormLabel>
              <FormControl>
                <Input placeholder="Enter crew name" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="description"
          render={({ field }) => (
            <FormItem>
              <FormLabel>
                Description
                <span className="text-gray-500 ml-1">(Optional)</span>
              </FormLabel>
              <FormControl>
                <Textarea
                  placeholder="Enter a brief description of the crew"
                  className="resize-none"
                  {...field}
                />
              </FormControl>
            </FormItem>
          )}
        />

        <div className="space-y-4">
          <h2 className="text-lg font-medium text-gray-900">Agents</h2>
          {fields.map((agent, index) => {
            const agentType = form.watch(
              `agents.${index}.agent_type`
            ) as AgentType;

            return (
              <div
                key={agent.id}
                className="border p-4 rounded-md mb-4 bg-gray-50"
                role="group"
                aria-label={`Agent ${index + 1}`}
              >
                <div className="flex gap-x-2 mb-4">
                  <FormField
                    control={form.control}
                    name={`agents.${index}.agent_type`}
                    render={({ field }) => (
                      <FormItem className="flex-grow">
                        <FormLabel>Agent Type</FormLabel>
                        <Select
                          onValueChange={field.onChange}
                          defaultValue={field.value}
                        >
                          <FormControl>
                            <SelectTrigger>
                              <SelectValue placeholder="Select agent type" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            <SelectItem value="solnai">Soln.ai</SelectItem>
                            <SelectItem value="autogen">AutoGen</SelectItem>
                          </SelectContent>
                        </Select>
                      </FormItem>
                    )}
                  />
                  <Button
                    type="button"
                    variant="destructive"
                    onClick={() => remove(index)}
                    className="self-end"
                    aria-label={`Remove Agent ${index + 1}`}
                  >
                    Remove
                  </Button>
                </div>

                <div className="space-y-4">
                  <FormField
                    control={form.control}
                    name={`agents.${index}.name`}
                    rules={{ required: "Agent name is required" }}
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Agent Name</FormLabel>
                        <FormControl>
                          <Input placeholder="Enter agent name" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name={`agents.${index}.role`}
                    rules={{ required: "Agent role is required" }}
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Role</FormLabel>
                        <FormControl>
                          <Input placeholder="Enter agent role" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  {agentType === "solnai" && (
                    <>
                      <FormField
                        control={form.control}
                        name={`agents.${index}.goal`}
                        rules={{ required: "Agent goal is required" }}
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Goal</FormLabel>
                            <FormControl>
                              <Textarea
                                placeholder="Enter agent goal"
                                className="resize-none"
                                {...field}
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name={`agents.${index}.backstory`}
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>
                              Backstory
                              <span className="text-gray-500 ml-1">
                                (Optional)
                              </span>
                            </FormLabel>
                            <FormControl>
                              <Textarea
                                placeholder="Enter agent backstory"
                                className="resize-none"
                                {...field}
                              />
                            </FormControl>
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name={`agents.${index}.llm`}
                        rules={{ required: "LLM is required" }}
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>LLM</FormLabel>
                            <Select
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                            >
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue placeholder="Select LLM" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="gpt-4-turbo-preview">
                                  GPT-4 Turbo
                                </SelectItem>
                                <SelectItem value="gpt-3.5-turbo-instruct">
                                  GPT-3.5 Turbo
                                </SelectItem>
                                <SelectItem value="o3-mini">o3-mini</SelectItem>
                                <SelectItem value="gemini-2.0-flash">
                                  Gemini 2.0 Flash
                                </SelectItem>
                                <SelectItem value="gemini-2.0-pro">
                                  Gemini 2.0 Pro
                                </SelectItem>
                              </SelectContent>
                            </Select>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    </>
                  )}

                  {agentType === "autogen" && (
                    <>
                      <FormField
                        control={form.control}
                        name={`agents.${index}.autogen_config.llm_config.temperature`}
                        rules={{
                          required: "Temperature is required",
                          min: {
                            value: 0,
                            message: "Should be between 0 and 1",
                          },
                          max: {
                            value: 1,
                            message: "Should be between 0 and 1",
                          },
                        }}
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Temperature</FormLabel>
                            <FormControl>
                              <Input
                                type="number"
                                step="0.1"
                                defaultValue={0.7}
                                {...field}
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name={`agents.${index}.autogen_config.llm_config.max_tokens`}
                        rules={{
                          required: "Max tokens is required",
                          min: { value: 1, message: "Should be more than 0" },
                        }}
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Max Tokens</FormLabel>
                            <FormControl>
                              <Input
                                type="number"
                                defaultValue={256}
                                {...field}
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name={`agents.${index}.autogen_config.llm_config.model`}
                        rules={{ required: "Model is required" }}
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Model</FormLabel>
                            <Select
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                            >
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue placeholder="Select model" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="gpt-4-turbo-preview">
                                  GPT-4 Turbo
                                </SelectItem>
                                <SelectItem value="gpt-3.5-turbo-instruct">
                                  GPT-3.5 Turbo
                                </SelectItem>
                                <SelectItem value="o3-mini">o3-mini</SelectItem>
                                <SelectItem value="gemini-2.0-flash">
                                  Gemini 2.0 Flash
                                </SelectItem>
                                <SelectItem value="gemini-2.0-pro">
                                  Gemini 2.0 Pro
                                </SelectItem>
                              </SelectContent>
                            </Select>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    </>
                  )}
                </div>
              </div>
            );
          })}
          <Button
            type="button"
            variant="secondary"
            onClick={() => append(defaultAgent())}
            className="w-full sm:w-auto"
            aria-label="Add New Agent"
          >
            Add Agent
          </Button>
        </div>

        <div className="flex justify-end">
          <Button
            type="submit"
            disabled={isLoading}
            className="w-full sm:w-auto"
            aria-disabled={isLoading}
          >
            {isLoading ? "Creating..." : "Create Crew"}
          </Button>
        </div>
      </form>
    </Form>
  );
};

export default CreateCrewForm;
