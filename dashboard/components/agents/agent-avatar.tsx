"use client";

import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

type Desk = "staff" | "spot" | "futures";
type Size = "sm" | "md" | "lg" | "xl";

interface AgentAvatarProps {
  name: string;
  desk: Desk;
  size?: Size;
  showTooltip?: boolean;
  className?: string;
}

// Get initials from agent name
function getInitials(name: string): string {
  const words = name.split(/[\s-]+/);
  if (words.length === 1) {
    return name.slice(0, 2).toUpperCase();
  }
  return words
    .slice(0, 2)
    .map((w) => w[0])
    .join("")
    .toUpperCase();
}

// Generate a consistent color based on name + desk
function getAgentColor(name: string, desk: Desk): { bg: string; text: string } {
  // Base hues for each desk
  const deskHues = {
    staff: 270,    // Purple
    spot: 30,      // Orange
    futures: 175,  // Teal
  };

  // Create a simple hash from the name for variation
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }

  // Vary the hue within a range around the desk's base hue
  const hueVariation = (hash % 30) - 15; // -15 to +15 variation
  const hue = deskHues[desk] + hueVariation;

  // Vary lightness slightly
  const lightnessVariation = (Math.abs(hash) % 15) - 7; // -7 to +7
  const lightness = 55 + lightnessVariation;

  // Saturation also varies slightly
  const saturation = 70 + (Math.abs(hash >> 4) % 20);

  return {
    bg: `hsl(${hue}, ${saturation}%, ${lightness}%)`,
    text: lightness > 60 ? "hsl(0, 0%, 10%)" : "hsl(0, 0%, 98%)",
  };
}

const sizeClasses: Record<Size, string> = {
  sm: "h-6 w-6 text-[10px]",
  md: "h-8 w-8 text-xs",
  lg: "h-10 w-10 text-sm",
  xl: "h-12 w-12 text-base",
};

export function AgentAvatar({
  name,
  desk,
  size = "md",
  showTooltip = true,
  className,
}: AgentAvatarProps) {
  const colors = getAgentColor(name, desk);
  const initials = getInitials(name);

  const avatar = (
    <Avatar
      className={cn(
        sizeClasses[size],
        "font-semibold transition-transform hover:scale-105",
        className
      )}
    >
      <AvatarFallback
        style={{
          backgroundColor: colors.bg,
          color: colors.text,
        }}
        className="font-semibold"
      >
        {initials}
      </AvatarFallback>
    </Avatar>
  );

  if (!showTooltip) {
    return avatar;
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>{avatar}</TooltipTrigger>
        <TooltipContent>
          <div className="flex items-center gap-2">
            <span className="font-medium">{name}</span>
            <span
              className={cn(
                "text-[10px] px-1.5 py-0.5 rounded uppercase",
                desk === "staff" && "bg-purple-500/20 text-purple-400",
                desk === "spot" && "bg-orange-500/20 text-orange-400",
                desk === "futures" && "bg-teal-500/20 text-teal-400"
              )}
            >
              {desk}
            </span>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// Export a group of avatars for showing multiple agents
interface AgentAvatarGroupProps {
  agents: Array<{ name: string; desk: Desk }>;
  max?: number;
  size?: Size;
}

export function AgentAvatarGroup({
  agents,
  max = 4,
  size = "sm",
}: AgentAvatarGroupProps) {
  const visibleAgents = agents.slice(0, max);
  const remainingCount = agents.length - max;

  return (
    <div className="flex -space-x-2">
      {visibleAgents.map((agent, index) => (
        <AgentAvatar
          key={`${agent.name}-${index}`}
          name={agent.name}
          desk={agent.desk}
          size={size}
          className="ring-2 ring-background"
        />
      ))}
      {remainingCount > 0 && (
        <div
          className={cn(
            sizeClasses[size],
            "flex items-center justify-center rounded-full bg-muted text-muted-foreground font-medium ring-2 ring-background"
          )}
        >
          +{remainingCount}
        </div>
      )}
    </div>
  );
}
