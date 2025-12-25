"use client";

import { useEffect, useRef, useMemo } from "react";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatMessage, CompactChatMessage } from "@/components/chat/chat-message";
import { MultiTypingIndicator } from "@/components/chat/typing-indicator";
import { useChatStore, type ChatMessage as ChatMessageType, type Desk } from "@/stores/chat-store";
import { Badge } from "@/components/ui/badge";
import { MessageSquare } from "lucide-react";

interface GroupedMessage {
  type: "message";
  message: ChatMessageType;
  isFirst: boolean;
  desk: Desk;
}

interface DateDivider {
  type: "divider";
  date: string;
}

type TimelineItem = GroupedMessage | DateDivider;

function groupMessages(messages: ChatMessageType[]): TimelineItem[] {
  const items: TimelineItem[] = [];
  let lastAgentId: string | null = null;
  let lastDate: string | null = null;

  messages.forEach((message) => {
    const messageDate = new Date(message.timestamp).toDateString();

    // Add date divider if day changed
    if (messageDate !== lastDate) {
      items.push({ type: "divider", date: messageDate });
      lastDate = messageDate;
      lastAgentId = null; // Reset agent grouping on new day
    }

    const isFirst = message.agentId !== lastAgentId;
    lastAgentId = message.agentId;

    items.push({
      type: "message",
      message,
      isFirst,
      desk: message.desk,
    });
  });

  return items;
}

interface ChatTimelineProps {
  className?: string;
}

export function ChatTimeline({ className }: ChatTimelineProps) {
  const { messages, typingAgentIds } = useChatStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const timelineItems = useMemo(() => groupMessages(messages), [messages]);

  const typingAgents = useMemo(() => {
    return Array.from(typingAgentIds).map((id) => {
      // Find the agent's last message to get their role and desk
      const lastMessage = [...messages]
        .reverse()
        .find((m) => m.agentId === id);
      return {
        name: lastMessage?.agentRole ?? id,
        desk: lastMessage?.desk ?? ("spot" as Desk),
      };
    });
  }, [typingAgentIds, messages]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  if (messages.length === 0) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center h-full text-muted-foreground",
          className
        )}
      >
        <MessageSquare className="h-12 w-12 mb-4 opacity-30" />
        <p className="text-sm font-medium">No messages yet</p>
        <p className="text-xs mt-1">Connect to a crew to see agent conversations</p>
      </div>
    );
  }

  return (
    <ScrollArea className={cn("flex-1", className)} ref={scrollRef}>
      <div className="py-4">
        {timelineItems.map((item, index) => {
          if (item.type === "divider") {
            return (
              <div
                key={`divider-${item.date}`}
                className="flex items-center gap-4 px-4 py-3"
              >
                <div className="flex-1 h-px bg-border" />
                <Badge variant="outline" className="text-xs font-normal">
                  {item.date}
                </Badge>
                <div className="flex-1 h-px bg-border" />
              </div>
            );
          }

          if (item.isFirst) {
            return (
              <ChatMessage
                key={item.message.id}
                message={item.message}
                desk={item.desk}
              />
            );
          }

          return (
            <CompactChatMessage key={item.message.id} message={item.message} />
          );
        })}

        {/* Typing Indicator */}
        {typingAgents.length > 0 && (
          <div className="px-4 py-3">
            <MultiTypingIndicator agents={typingAgents} />
          </div>
        )}

        {/* Scroll anchor */}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
