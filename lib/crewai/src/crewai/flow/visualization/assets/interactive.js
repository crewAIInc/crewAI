"use strict";

const CONSTANTS = {
  NODE: {
    BASE_WIDTH: 220,
    BASE_HEIGHT: 100,
    BORDER_RADIUS: 20,
    TEXT_SIZE: 13,
    TEXT_PADDING: 16,
    TEXT_BG_RADIUS: 6,
    HOVER_SCALE: 1.00,
    PRESSED_SCALE: 1.16,
    SELECTED_SCALE: 1.05,
  },
  EDGE: {
    DEFAULT_WIDTH: 2,
    HIGHLIGHTED_WIDTH: 8,
    ANIMATION_DURATION: 300,
    DEFAULT_SHADOW_SIZE: 4,
    HIGHLIGHTED_SHADOW_SIZE: 20,
  },
  ANIMATION: {
    DURATION: 300,
    EASE_OUT_CUBIC: (t) => 1 - Math.pow(1 - t, 3),
  },
  NETWORK: {
    STABILIZATION_ITERATIONS: 300,
    NODE_DISTANCE: 225,
    SPRING_LENGTH: 100,
    LEVEL_SEPARATION: 150,
    NODE_SPACING: 350,
    TREE_SPACING: 250,
  },
  DRAWER: {
    WIDTH: 400,
  },
};

function loadVisCDN() {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js";
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

function drawRoundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function highlightPython(code) {
  return Prism.highlight(code, Prism.languages.python, "python");
}

class NodeRenderer {
  constructor(nodes, networkManager) {
    this.nodes = nodes;
    this.networkManager = networkManager;
    this.nodeScales = new Map();
    this.scaleAnimations = new Map();
    this.hoverGlowIntensities = new Map();
    this.glowAnimations = new Map();
    this.colorCache = new Map();
    this.tempCanvas = document.createElement('canvas');
    this.tempCanvas.width = 1;
    this.tempCanvas.height = 1;
    this.tempCtx = this.tempCanvas.getContext('2d');
  }

  render({ ctx, id, x, y }) {
    const node = this.nodes.get(id);
    if (!node?.nodeStyle) return {};

    const scale = this.getNodeScale(id);
    const isActiveDrawer = this.networkManager.drawerManager?.activeNodeId === id;
    const isHovered = this.networkManager.hoveredNodeId === id && !isActiveDrawer;
    const nodeStyle = node.nodeStyle;

    // Manage hover glow intensity animation
    const glowIntensity = this.getHoverGlowIntensity(id, isHovered);

    ctx.font = `500 ${CONSTANTS.NODE.TEXT_SIZE * scale}px 'JetBrains Mono', 'SF Mono', 'Monaco', 'Menlo', 'Consolas', monospace`;
    const textMetrics = ctx.measureText(nodeStyle.name);
    const textWidth = textMetrics.width;
    const textHeight = CONSTANTS.NODE.TEXT_SIZE * scale;
    const textPadding = CONSTANTS.NODE.TEXT_PADDING * scale;

    const width = textWidth + textPadding * 5;
    const height = textHeight + textPadding * 2.5;

    return {
      drawNode: () => {
        ctx.save();
        const opacity = node.opacity !== undefined ? node.opacity : 1.0;
        this.applyShadow(ctx, node, glowIntensity, opacity);
        ctx.globalAlpha = opacity;
        this.drawNodeShape(ctx, x, y, width, height, scale, nodeStyle, opacity, node);
        this.drawNodeText(ctx, x, y, scale, nodeStyle, opacity, node);
        ctx.restore();
      },
      nodeDimensions: { width, height },
    };
  }

  getNodeScale(id) {
    const isActiveDrawer = this.networkManager.drawerManager?.activeNodeId === id;

    let targetScale = 1.0;
    if (isActiveDrawer) {
      targetScale = CONSTANTS.NODE.SELECTED_SCALE;
    } else if (this.networkManager.pressedNodeId === id) {
      targetScale = CONSTANTS.NODE.PRESSED_SCALE;
    } else if (this.networkManager.hoveredNodeId === id) {
      targetScale = CONSTANTS.NODE.HOVER_SCALE;
    }

    const currentScale = this.nodeScales.get(id) ?? 1.0;
    const runningAnimation = this.scaleAnimations.get(id);
    const animationTarget = runningAnimation?.targetScale;

    if (Math.abs(targetScale - currentScale) > 0.001) {
      if (runningAnimation && animationTarget !== targetScale) {
        cancelAnimationFrame(runningAnimation.frameId);
        this.scaleAnimations.delete(id);
      }

      if (!this.scaleAnimations.has(id)) {
        this.animateScale(id, currentScale, targetScale);
      }
    }

    return currentScale;
  }

  animateScale(id, startScale, targetScale) {
    const startTime = performance.now();
    const duration = 150;

    const animate = () => {
      const elapsed = performance.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = CONSTANTS.ANIMATION.EASE_OUT_CUBIC(progress);

      const currentScale = startScale + (targetScale - startScale) * eased;
      this.nodeScales.set(id, currentScale);

      if (progress < 1) {
        const frameId = requestAnimationFrame(animate);
        this.scaleAnimations.set(id, { frameId, targetScale });
      } else {
        this.scaleAnimations.delete(id);
        this.nodeScales.set(id, targetScale);
      }

      this.networkManager.network?.redraw();
    };

    animate();
  }

  getHoverGlowIntensity(id, isHovered) {
    const targetIntensity = isHovered ? 1.0 : 0.0;
    const currentIntensity = this.hoverGlowIntensities.get(id) ?? 0.0;
    const runningAnimation = this.glowAnimations.get(id);
    const animationTarget = runningAnimation?.targetIntensity;

    if (Math.abs(targetIntensity - currentIntensity) > 0.001) {
      if (runningAnimation && animationTarget !== targetIntensity) {
        cancelAnimationFrame(runningAnimation.frameId);
        this.glowAnimations.delete(id);
      }

      if (!this.glowAnimations.has(id)) {
        this.animateGlowIntensity(id, currentIntensity, targetIntensity);
      }
    }

    return currentIntensity;
  }

  animateGlowIntensity(id, startIntensity, targetIntensity) {
    const startTime = performance.now();
    const duration = 200;

    const animate = () => {
      const elapsed = performance.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = CONSTANTS.ANIMATION.EASE_OUT_CUBIC(progress);

      const currentIntensity = startIntensity + (targetIntensity - startIntensity) * eased;
      this.hoverGlowIntensities.set(id, currentIntensity);

      if (progress < 1) {
        const frameId = requestAnimationFrame(animate);
        this.glowAnimations.set(id, { frameId, targetIntensity });
      } else {
        this.glowAnimations.delete(id);
        this.hoverGlowIntensities.set(id, targetIntensity);
      }

      this.networkManager.network?.redraw();
    };

    animate();
  }

  applyShadow(ctx, node, glowIntensity = 0, nodeOpacity = 1.0) {
    if (glowIntensity > 0.001) {
      // Save current alpha and apply glow at full opacity
      const currentAlpha = ctx.globalAlpha;
      ctx.globalAlpha = 1.0;

      const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';

      // Use CrewAI orange for hover glow in both themes
      const glowR = 255;
      const glowG = 90;
      const glowB = 80;
      const blurRadius = isDarkMode ? 20 : 35;

      // Scale glow intensity proportionally based on node opacity
      // When node is inactive (opacity < 1.0), reduce glow intensity accordingly
      const scaledGlowIntensity = glowIntensity * nodeOpacity;

      const glowColor = `rgba(${glowR}, ${glowG}, ${glowB}, ${scaledGlowIntensity})`;

      ctx.shadowColor = glowColor;
      ctx.shadowBlur = blurRadius * scaledGlowIntensity;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;

      // Restore the original alpha
      ctx.globalAlpha = currentAlpha;
      return;
    }

    if (node.shadow?.enabled) {
      ctx.shadowColor = node.shadow.color || "rgba(0,0,0,0.1)";
      ctx.shadowBlur = node.shadow.size || 8;
      ctx.shadowOffsetX = node.shadow.x || 0;
      ctx.shadowOffsetY = node.shadow.y || 0;
      return;
    }

    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
  }

  resolveCSSVariable(color) {
    if (color?.startsWith('var(')) {
      const varName = color.match(/var\((--[^)]+)\)/)?.[1];
      if (varName) {
        return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
      }
    }
    return color;
  }


  parseColor(color) {
    const cacheKey = `parse_${color}`;
    if (this.colorCache.has(cacheKey)) {
      return this.colorCache.get(cacheKey);
    }

    this.tempCtx.fillStyle = color;
    this.tempCtx.fillRect(0, 0, 1, 1);
    const [r, g, b] = this.tempCtx.getImageData(0, 0, 1, 1).data;

    const result = { r, g, b };
    this.colorCache.set(cacheKey, result);
    return result;
  }

  darkenColor(color, opacity) {
    if (opacity >= 0.9) return color;

    const { r, g, b } = this.parseColor(color);

    const t = (opacity - 0.85) / (1.0 - 0.85);
    const normalizedT = Math.max(0, Math.min(1, t));

    const minBrightness = 0.4;
    const brightness = minBrightness + (1.0 - minBrightness) * normalizedT;

    const newR = Math.floor(r * brightness);
    const newG = Math.floor(g * brightness);
    const newB = Math.floor(b * brightness);

    return `rgb(${newR}, ${newG}, ${newB})`;
  }

  desaturateColor(color, opacity) {
    if (opacity >= 0.9) return color;

    const { r, g, b } = this.parseColor(color);

    // Convert to HSL to adjust saturation and lightness
    const max = Math.max(r, g, b) / 255;
    const min = Math.min(r, g, b) / 255;
    const l = (max + min) / 2;
    let h = 0, s = 0;

    if (max !== min) {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

      if (max === r / 255) {
        h = ((g / 255 - b / 255) / d + (g < b ? 6 : 0)) / 6;
      } else if (max === g / 255) {
        h = ((b / 255 - r / 255) / d + 2) / 6;
      } else {
        h = ((r / 255 - g / 255) / d + 4) / 6;
      }
    }

    // Reduce saturation and lightness by 40%
    s = s * 0.6;
    const newL = l * 0.6;

    // Convert back to RGB
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };

    let newR, newG, newB;
    if (s === 0) {
      newR = newG = newB = Math.floor(newL * 255);
    } else {
      const q = newL < 0.5 ? newL * (1 + s) : newL + s - newL * s;
      const p = 2 * newL - q;
      newR = Math.floor(hue2rgb(p, q, h + 1/3) * 255);
      newG = Math.floor(hue2rgb(p, q, h) * 255);
      newB = Math.floor(hue2rgb(p, q, h - 1/3) * 255);
    }

    return `rgb(${newR}, ${newG}, ${newB})`;
  }

  drawNodeShape(ctx, x, y, width, height, scale, nodeStyle, opacity = 1.0, node = null) {
    const radius = CONSTANTS.NODE.BORDER_RADIUS * scale;
    const rectX = x - width / 2;
    const rectY = y - height / 2;

    const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
    const nodeData = '{{ nodeData }}';
    const metadata = node ? nodeData[node.id] : null;
    const isStartNode = metadata && metadata.type === 'start';

    let nodeColor;

    if (isDarkMode || isStartNode) {
      // In dark mode or for start nodes, use the theme color
      nodeColor = this.resolveCSSVariable(nodeStyle.bgColor);
    } else {
      // In light mode for non-start nodes, use white
      nodeColor = 'rgb(255, 255, 255)';
    }

    // Parse the base color to get RGB values
    let { r, g, b } = this.parseColor(nodeColor);

    // For inactive nodes, check if node is in highlighted list
    // If drawer is open and node is not highlighted, it's inactive
    const isDrawerOpen = this.networkManager.drawerManager?.activeNodeId !== null;
    const isHighlighted = this.networkManager.triggeredByHighlighter?.highlightedNodes?.includes(node?.id);
    const isActiveNode = this.networkManager.drawerManager?.activeNodeId === node?.id;
    const hasHighlightedNodes = this.networkManager.triggeredByHighlighter?.highlightedNodes?.length > 0;

    // Non-prominent nodes: drawer is open, has highlighted nodes, but this node is not highlighted or active
    const isNonProminent = isDrawerOpen && hasHighlightedNodes && !isHighlighted && !isActiveNode;

    // Inactive nodes: drawer is open but no highlighted nodes, and this node is not active
    const isInactive = isDrawerOpen && !hasHighlightedNodes && !isActiveNode;

    if (isNonProminent || isInactive) {
      // Make non-prominent and inactive nodes a darker version of the normal active color
      const darkenFactor = 0.4; // Keep 40% of original color (darken by 60%)
      r = Math.round(r * darkenFactor);
      g = Math.round(g * darkenFactor);
      b = Math.round(b * darkenFactor);
    }

    // Draw base shape with frosted glass effect
    ctx.beginPath();
    drawRoundedRect(ctx, rectX, rectY, width, height, radius);
    // Use full opacity for all nodes
    const glassOpacity = 1.0;
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${glassOpacity})`;
    ctx.fill();

    // Calculate text label area to exclude from frosted overlay
    const textPadding = CONSTANTS.NODE.TEXT_PADDING * scale;
    const textBgRadius = CONSTANTS.NODE.TEXT_BG_RADIUS * scale;

    ctx.font = `500 ${CONSTANTS.NODE.TEXT_SIZE * scale}px 'JetBrains Mono', 'SF Mono', 'Monaco', 'Menlo', 'Consolas', monospace`;
    const textMetrics = ctx.measureText(nodeStyle.name);
    const textWidth = textMetrics.width;
    const textHeight = CONSTANTS.NODE.TEXT_SIZE * scale;
    const textBgWidth = textWidth + textPadding * 2;
    const textBgHeight = textHeight + textPadding * 0.75;
    const textBgX = x - textBgWidth / 2;
    const textBgY = y - textBgHeight / 2;

    // Add frosted overlay (clipped to node shape, excluding text area)
    ctx.save();
    ctx.beginPath();
    drawRoundedRect(ctx, rectX, rectY, width, height, radius);
    ctx.clip();

    // Cut out the text label area from the frosted overlay
    ctx.beginPath();
    drawRoundedRect(ctx, rectX, rectY, width, height, radius);
    drawRoundedRect(ctx, textBgX, textBgY, textBgWidth, textBgHeight, textBgRadius);
    ctx.clip('evenodd');

    // For inactive nodes, use stronger absolute frost values
    // For active nodes, scale frost with opacity
    let frostTop, frostMid, frostBottom;
    if (isInactive) {
      // Inactive nodes get stronger, more consistent frost
      frostTop = 0.45;
      frostMid = 0.35;
      frostBottom = 0.25;
    } else {
      // Active nodes get opacity-scaled frost
      frostTop = opacity * 0.3;
      frostMid = opacity * 0.2;
      frostBottom = opacity * 0.15;
    }

    // Stronger white overlay for frosted appearance
    const frostOverlay = ctx.createLinearGradient(rectX, rectY, rectX, rectY + height);
    frostOverlay.addColorStop(0, `rgba(255, 255, 255, ${frostTop})`);
    frostOverlay.addColorStop(0.5, `rgba(255, 255, 255, ${frostMid})`);
    frostOverlay.addColorStop(1, `rgba(255, 255, 255, ${frostBottom})`);

    ctx.fillStyle = frostOverlay;
    ctx.fillRect(rectX, rectY, width, height);
    ctx.restore();

    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;

    // Draw border at full opacity (desaturated for inactive nodes)
    // Reset globalAlpha to 1.0 so the border is always fully visible
    ctx.save();
    ctx.globalAlpha = 1.0;
    ctx.beginPath();
    drawRoundedRect(ctx, rectX, rectY, width, height, radius);
    const borderColor = this.resolveCSSVariable(nodeStyle.borderColor);
    let finalBorderColor = this.desaturateColor(borderColor, opacity);

    // Darken border color for non-prominent and inactive nodes
    if (isNonProminent || isInactive) {
      const borderRGB = this.parseColor(finalBorderColor);
      const darkenFactor = 0.4;
      const darkenedR = Math.round(borderRGB.r * darkenFactor);
      const darkenedG = Math.round(borderRGB.g * darkenFactor);
      const darkenedB = Math.round(borderRGB.b * darkenFactor);
      finalBorderColor = `rgb(${darkenedR}, ${darkenedG}, ${darkenedB})`;
    }

    ctx.strokeStyle = finalBorderColor;
    ctx.lineWidth = nodeStyle.borderWidth * scale;
    ctx.stroke();
    ctx.restore();
  }

  drawNodeText(ctx, x, y, scale, nodeStyle, opacity = 1.0, node = null) {
    ctx.font = `500 ${CONSTANTS.NODE.TEXT_SIZE * scale}px 'JetBrains Mono', 'SF Mono', 'Monaco', 'Menlo', 'Consolas', monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    const textMetrics = ctx.measureText(nodeStyle.name);
    const textWidth = textMetrics.width;
    const textHeight = CONSTANTS.NODE.TEXT_SIZE * scale;
    const textPadding = CONSTANTS.NODE.TEXT_PADDING * scale;
    const textBgRadius = CONSTANTS.NODE.TEXT_BG_RADIUS * scale;

    const textBgWidth = textWidth + textPadding * 2;
    const textBgHeight = textHeight + textPadding * 0.75;
    const textBgX = x - textBgWidth / 2;
    const textBgY = y - textBgHeight / 2;

    drawRoundedRect(
      ctx,
      textBgX,
      textBgY,
      textBgWidth,
      textBgHeight,
      textBgRadius,
    );

    const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
    const nodeData = '{{ nodeData }}';
    const metadata = node ? nodeData[node.id] : null;
    const isStartNode = metadata && metadata.type === 'start';

    // Check if this is an inactive or non-prominent node using the same logic as drawNodeShape
    const isDrawerOpen = this.networkManager.drawerManager?.activeNodeId !== null;
    const isHighlighted = this.networkManager.triggeredByHighlighter?.highlightedNodes?.includes(node?.id);
    const isActiveNode = this.networkManager.drawerManager?.activeNodeId === node?.id;
    const hasHighlightedNodes = this.networkManager.triggeredByHighlighter?.highlightedNodes?.length > 0;

    const isNonProminent = isDrawerOpen && hasHighlightedNodes && !isHighlighted && !isActiveNode;
    const isInactive = isDrawerOpen && !hasHighlightedNodes && !isActiveNode;

    // Get the base node color to darken it for inactive nodes
    let nodeColor;
    if (isDarkMode || isStartNode) {
      nodeColor = this.resolveCSSVariable(nodeStyle.bgColor);
    } else {
      nodeColor = 'rgb(255, 255, 255)';
    }
    const { r, g, b } = this.parseColor(nodeColor);

    let labelBgR = 255, labelBgG = 255, labelBgB = 255;
    let labelBgOpacity = 0.2 * opacity;

    if (isNonProminent || isInactive) {
      // Darken the base node color for non-prominent and inactive label backgrounds
      const darkenFactor = 0.4;
      labelBgR = Math.round(r * darkenFactor);
      labelBgG = Math.round(g * darkenFactor);
      labelBgB = Math.round(b * darkenFactor);
      labelBgOpacity = 0.5;
    } else if (!isDarkMode && !isStartNode) {
      // In light mode for non-start nodes, use gray for active node labels
      labelBgR = labelBgG = labelBgB = 128;
      labelBgOpacity = 0.25;
    }

    ctx.fillStyle = `rgba(${labelBgR}, ${labelBgG}, ${labelBgB}, ${labelBgOpacity})`;
    ctx.fill();

    // For start nodes or dark mode, use theme color; in light mode, use dark text
    let fontColor;
    if (isDarkMode || isStartNode) {
      fontColor = this.resolveCSSVariable(nodeStyle.fontColor);
    } else {
      fontColor = 'rgb(30, 30, 30)';
    }

    // Darken font color for non-prominent and inactive nodes
    if (isNonProminent || isInactive) {
      const fontRGB = this.parseColor(fontColor);
      const darkenFactor = 0.4;
      const darkenedR = Math.round(fontRGB.r * darkenFactor);
      const darkenedG = Math.round(fontRGB.g * darkenFactor);
      const darkenedB = Math.round(fontRGB.b * darkenFactor);
      fontColor = `rgb(${darkenedR}, ${darkenedG}, ${darkenedB})`;
    }

    ctx.fillStyle = fontColor;
    ctx.fillText(nodeStyle.name, x, y);
  }
}

class AnimationManager {
  constructor() {
    this.animations = new Map();
  }

  animateEdgeWidth(
    edges,
    edgeId,
    targetWidth,
    duration = CONSTANTS.EDGE.ANIMATION_DURATION,
  ) {
    this.cancel(edgeId);

    const edge = edges.get(edgeId);
    if (!edge) return;

    const startWidth = edge.width || CONSTANTS.EDGE.DEFAULT_WIDTH;
    const startTime = performance.now();

    const animate = () => {
      const elapsed = performance.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = CONSTANTS.ANIMATION.EASE_OUT_CUBIC(progress);
      const currentWidth = startWidth + (targetWidth - startWidth) * eased;

      edges.update({ id: edgeId, width: currentWidth });

      if (progress < 1) {
        const frameId = requestAnimationFrame(animate);
        this.animations.set(edgeId, frameId);
      } else {
        this.animations.delete(edgeId);
      }
    };

    animate();
  }

  cancel(id) {
    if (this.animations.has(id)) {
      cancelAnimationFrame(this.animations.get(id));
      this.animations.delete(id);
    }
  }

  cancelAll() {
    this.animations.forEach((frameId) => cancelAnimationFrame(frameId));
    this.animations.clear();
  }
}

class TriggeredByHighlighter {
  constructor(network, nodes, edges, highlightCanvas) {
    this.network = network;
    this.nodes = nodes;
    this.edges = edges;
    this.canvas = highlightCanvas;
    this.ctx = highlightCanvas.getContext("2d");

    this.highlightedNodes = [];
    this.highlightedEdges = [];
    this.activeDrawerNodeId = null;
    this.activeDrawerEdges = [];

    this.setupCanvas();
  }

  setupCanvas() {
    this.resizeCanvas();
    this.canvas.classList.remove("visible");
    window.addEventListener("resize", () => this.resizeCanvas());
  }

  resizeCanvas() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  setActiveDrawer(nodeId, edges) {
    this.activeDrawerNodeId = nodeId;
    this.activeDrawerEdges = edges;
  }

  highlightTriggeredByGroup(triggerNodeIds) {
    this.clear();

    if (!this.activeDrawerNodeId || !triggerNodeIds || triggerNodeIds.length === 0) {
      return;
    }

    const allEdges = this.edges.get();
    const pathNodes = new Set([this.activeDrawerNodeId]);
    const connectingEdges = [];
    const nodeData = '{{ nodeData }}';

    triggerNodeIds.forEach(triggerNodeId => {
      const directEdges = allEdges.filter(
        (edge) => edge.from === triggerNodeId && edge.to === this.activeDrawerNodeId
      );

      if (directEdges.length > 0) {
        directEdges.forEach(edge => {
          connectingEdges.push(edge);
          pathNodes.add(edge.from);
          pathNodes.add(edge.to);
        });
      } else {
        for (const [nodeName, nodeInfo] of Object.entries(nodeData)) {
          if (nodeInfo.router_paths && nodeInfo.router_paths.includes(triggerNodeId)) {
            const routerNode = nodeName;

            const routerEdges = allEdges.filter(
              (edge) => edge.from === routerNode && edge.dashes
            );
            let foundEdge = false;

            for (const routerEdge of routerEdges) {
              if (routerEdge.label === triggerNodeId) {
                connectingEdges.push(routerEdge);
                pathNodes.add(routerNode);
                pathNodes.add(routerEdge.to);

                if (routerEdge.to !== this.activeDrawerNodeId) {
                  const pathToActive = allEdges.filter(
                    (edge) => edge.from === routerEdge.to && edge.to === this.activeDrawerNodeId
                  );

                  if (pathToActive.length > 0) {
                    connectingEdges.push(...pathToActive);
                    pathNodes.add(this.activeDrawerNodeId);
                  }
                }

                foundEdge = true;
                break;
              }
            }

            if (!foundEdge) {
              for (const routerEdge of routerEdges) {
                if (routerEdge.to === triggerNodeId) {
                  connectingEdges.push(routerEdge);
                  pathNodes.add(routerNode);
                  pathNodes.add(routerEdge.to);

                  const pathToActive = allEdges.filter(
                    (edge) => edge.from === triggerNodeId && edge.to === this.activeDrawerNodeId
                  );

                  if (pathToActive.length > 0) {
                    connectingEdges.push(...pathToActive);
                    pathNodes.add(this.activeDrawerNodeId);
                  }

                  foundEdge = true;
                  break;
                }
              }
            }

            if (!foundEdge) {
              const directRouterEdge = routerEdges.find(
                (edge) => edge.to === this.activeDrawerNodeId
              );

              if (directRouterEdge) {
                connectingEdges.push(directRouterEdge);
                pathNodes.add(routerNode);
                pathNodes.add(this.activeDrawerNodeId);
                foundEdge = true;
              }
            }

            if (foundEdge) {
              break;
            }
          }
        }
      }
    });

    if (connectingEdges.length === 0) {
      return;
    }

    this.highlightedNodes = Array.from(pathNodes);
    this.highlightedEdges = connectingEdges.map((e) => e.id);

    this.animateNodeOpacity();
    this.animateEdgeStyles();
  }

  highlightAllRouterPaths() {
    this.clear();

    if (!this.activeDrawerNodeId) {
      return;
    }

    const allEdges = this.edges.get();
    const nodeData = '{{ nodeData }}';
    const activeMetadata = nodeData[this.activeDrawerNodeId];

    const outgoingRouterEdges = allEdges.filter(
      (edge) => edge.from === this.activeDrawerNodeId && edge.dashes
    );

    let routerEdges = [];
    const pathNodes = new Set();

    if (outgoingRouterEdges.length > 0) {
      routerEdges = outgoingRouterEdges;
      pathNodes.add(this.activeDrawerNodeId);
      routerEdges.forEach(edge => {
        pathNodes.add(edge.to);
      });
    } else if (activeMetadata && activeMetadata.router_paths && activeMetadata.router_paths.length > 0) {
      activeMetadata.router_paths.forEach(pathName => {
        for (const [nodeName, nodeInfo] of Object.entries(nodeData)) {
          if (nodeInfo.router_paths && nodeInfo.router_paths.includes(pathName)) {
            const edgeFromRouter = allEdges.filter(
              (edge) => edge.from === nodeName && edge.to === this.activeDrawerNodeId && edge.dashes
            );

            if (edgeFromRouter.length > 0) {
              routerEdges.push(...edgeFromRouter);
              pathNodes.add(nodeName);
              pathNodes.add(this.activeDrawerNodeId);
            }
          }
        }
      });
    }

    if (routerEdges.length === 0) {
      return;
    }

    this.highlightedNodes = Array.from(pathNodes);
    this.highlightedEdges = routerEdges.map((e) => e.id);

    this.animateNodeOpacity();
    this.animateEdgeStyles();
  }

  highlightTriggeredBy(triggerNodeId) {
    this.clear();

    if (this.activeDrawerEdges && this.activeDrawerEdges.length > 0) {
      // Animate the activeDrawerEdges back to default
      this.resetEdgesToDefault(this.activeDrawerEdges);
      this.activeDrawerEdges = [];
    }

    if (!this.activeDrawerNodeId || !triggerNodeId) {
      return;
    }

    const allEdges = this.edges.get();
    let connectingEdges = [];
    let actualTriggerNodeId = triggerNodeId;

    connectingEdges = allEdges.filter(
      (edge) =>
        edge.from === triggerNodeId && edge.to === this.activeDrawerNodeId,
    );

    if (connectingEdges.length === 0) {
      const incomingRouterEdges = allEdges.filter(
        (edge) => edge.to === this.activeDrawerNodeId && edge.dashes,
      );

      if (incomingRouterEdges.length > 0) {
        incomingRouterEdges.forEach((edge) => {
          connectingEdges.push(edge);
          actualTriggerNodeId = edge.from;
        });
      }
    }

    if (connectingEdges.length === 0) {
      const outgoingRouterEdges = allEdges.filter(
        (edge) => edge.from === this.activeDrawerNodeId && edge.dashes,
      );

      if (outgoingRouterEdges.length > 0) {
        const nodeData = '{{ nodeData }}';
        for (const [nodeName, nodeInfo] of Object.entries(nodeData)) {
          if (
            nodeInfo.trigger_methods &&
            nodeInfo.trigger_methods.includes(triggerNodeId)
          ) {
            const edgeToTarget = outgoingRouterEdges.find(
              (e) => e.to === nodeName,
            );
            if (edgeToTarget) {
              connectingEdges.push(edgeToTarget);
              actualTriggerNodeId = nodeName;
              break;
            }
          }
        }
      }
    }

    if (connectingEdges.length === 0) {
      const nodeData = '{{ nodeData }}';

      const activeMetadata = nodeData[this.activeDrawerNodeId];
      if (
        activeMetadata &&
        activeMetadata.trigger_methods &&
        activeMetadata.trigger_methods.includes(triggerNodeId)
      ) {
        for (const [nodeName, nodeInfo] of Object.entries(nodeData)) {
          if (
            nodeInfo.router_paths &&
            nodeInfo.router_paths.includes(triggerNodeId)
          ) {
            const routerNode = nodeName;

            const routerEdges = allEdges.filter(
              (edge) => edge.from === routerNode && edge.dashes,
            );

            for (const routerEdge of routerEdges) {
              const intermediateNode = routerEdge.to;

              if (intermediateNode === this.activeDrawerNodeId) {
                connectingEdges = [routerEdge];
                actualTriggerNodeId = routerNode;
                break;
              }

              const pathToActive = allEdges.filter(
                (edge) =>
                  edge.from === intermediateNode &&
                  edge.to === this.activeDrawerNodeId,
              );

              if (pathToActive.length > 0) {
                connectingEdges = [routerEdge, ...pathToActive];
                actualTriggerNodeId = routerNode;
                break;
              }
            }

            if (connectingEdges.length > 0) break;
          }
        }
      }
    }

    if (connectingEdges.length === 0) {
      const edgesWithLabel = allEdges.filter(
        (edge) =>
          edge.dashes &&
          edge.label === triggerNodeId &&
          edge.to === this.activeDrawerNodeId,
      );

      if (edgesWithLabel.length > 0) {
        connectingEdges = edgesWithLabel;
        const firstEdge = edgesWithLabel[0];
        actualTriggerNodeId = firstEdge.from;
      }
    }

    if (connectingEdges.length === 0) {
      return;
    }

    const pathNodes = new Set([actualTriggerNodeId, this.activeDrawerNodeId]);
    connectingEdges.forEach((edge) => {
      pathNodes.add(edge.from);
      pathNodes.add(edge.to);
    });

    this.highlightedNodes = Array.from(pathNodes);
    this.highlightedEdges = connectingEdges.map((e) => e.id);

    this.animateNodeOpacity();
    this.animateEdgeStyles();
  }

  animateNodeOpacity() {
    const allNodesList = this.nodes.get();
    const nodeAnimDuration = CONSTANTS.ANIMATION.DURATION;
    const nodeAnimStart = performance.now();
    const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';

    const animate = () => {
      const elapsed = performance.now() - nodeAnimStart;
      const progress = Math.min(elapsed / nodeAnimDuration, 1);
      const eased = CONSTANTS.ANIMATION.EASE_OUT_CUBIC(progress);

      allNodesList.forEach((node) => {
        const currentOpacity = node.opacity !== undefined ? node.opacity : 1.0;
        // Keep inactive nodes at full opacity
        const inactiveOpacity = 1.0;
        const targetOpacity = this.highlightedNodes.includes(node.id)
          ? 1.0
          : inactiveOpacity;
        const newOpacity =
          currentOpacity + (targetOpacity - currentOpacity) * eased;

        this.nodes.update({
          id: node.id,
          opacity: newOpacity,
        });
      });

      this.network.redraw();

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    animate();
  }

  animateEdgeStyles() {
    const edgeIds = this.edges.getIds();
    const edgeAnimDuration = CONSTANTS.ANIMATION.DURATION;
    const edgeAnimStart = performance.now();

    const animate = () => {
      const elapsed = performance.now() - edgeAnimStart;
      const progress = Math.min(elapsed / edgeAnimDuration, 1);
      const eased = CONSTANTS.ANIMATION.EASE_OUT_CUBIC(progress);

      edgeIds.forEach((edgeId) => {
        const edge = this.edges.get(edgeId);
        if (!edge) return;

        if (this.highlightedEdges.includes(edge.id)) {
          const currentWidth = edge.width || CONSTANTS.EDGE.DEFAULT_WIDTH;
          const targetWidth = CONSTANTS.EDGE.HIGHLIGHTED_WIDTH;
          const newWidth = currentWidth + (targetWidth - currentWidth) * eased;

          const currentShadowSize =
            edge.shadow?.size || CONSTANTS.EDGE.DEFAULT_SHADOW_SIZE;
          const targetShadowSize = CONSTANTS.EDGE.HIGHLIGHTED_SHADOW_SIZE;
          const newShadowSize =
            currentShadowSize + (targetShadowSize - currentShadowSize) * eased;

          const isAndOrRouter = edge.dashes || edge.label === "AND";
          const highlightColor = isAndOrRouter
            ? "{{ CREWAI_ORANGE }}"
            : getComputedStyle(document.documentElement).getPropertyValue('--edge-or-color').trim();

          const updateData = {
            id: edge.id,
            hidden: false,
            opacity: 1.0,
            width: newWidth,
            color: {
              color: highlightColor,
              highlight: highlightColor,
            },
            shadow: {
              enabled: true,
              color: highlightColor,
              size: newShadowSize,
              x: 0,
              y: 0,
            },
          };

          if (edge.dashes) {
            const scale = Math.sqrt(newWidth / CONSTANTS.EDGE.DEFAULT_WIDTH);
            updateData.dashes = [15 * scale, 10 * scale];
          }

          updateData.arrows = {
            to: {
              enabled: true,
              scaleFactor: 0.8,
              type: "triangle",
            },
          };

          updateData.color = {
            color: highlightColor,
            highlight: highlightColor,
            hover: highlightColor,
            inherit: "to",
          };

          this.edges.update(updateData);
        } else {
          const currentOpacity = edge.opacity !== undefined ? edge.opacity : 1.0;
          // Keep inactive edges at full opacity
          const targetOpacity = 1.0;
          const newOpacity = currentOpacity + (targetOpacity - currentOpacity) * eased;

          const currentWidth = edge.width !== undefined ? edge.width : CONSTANTS.EDGE.DEFAULT_WIDTH;
          const targetWidth = 1.2;
          const newWidth = currentWidth + (targetWidth - currentWidth) * eased;

          // Keep the original edge color instead of turning gray
          const isAndOrRouter = edge.dashes || edge.label === "AND";
          const baseColor = isAndOrRouter
            ? "{{ CREWAI_ORANGE }}"
            : getComputedStyle(document.documentElement).getPropertyValue('--edge-or-color').trim();

          // Convert color to rgba with opacity for vis.js
          let inactiveEdgeColor;
          if (baseColor.startsWith('#')) {
            // Convert hex to rgba
            const hex = baseColor.replace('#', '');
            const r = parseInt(hex.substr(0, 2), 16);
            const g = parseInt(hex.substr(2, 2), 16);
            const b = parseInt(hex.substr(4, 2), 16);
            inactiveEdgeColor = `rgba(${r}, ${g}, ${b}, ${newOpacity})`;
          } else if (baseColor.startsWith('rgb(')) {
            inactiveEdgeColor = baseColor.replace('rgb(', `rgba(`).replace(')', `, ${newOpacity})`);
          } else {
            inactiveEdgeColor = baseColor;
          }

          this.edges.update({
            id: edge.id,
            hidden: false,
            width: newWidth,
            color: {
              color: inactiveEdgeColor,
              highlight: inactiveEdgeColor,
              hover: inactiveEdgeColor,
            },
            shadow: {
              enabled: false,
            },
          });
        }
      });

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    animate();
  }

  resetEdgesToDefault(edgeIds = null, excludeEdges = []) {
    const targetEdgeIds = edgeIds || this.edges.getIds();
    const edgeAnimDuration = CONSTANTS.ANIMATION.DURATION;
    const edgeAnimStart = performance.now();

    const animate = () => {
      const elapsed = performance.now() - edgeAnimStart;
      const progress = Math.min(elapsed / edgeAnimDuration, 1);
      const eased = CONSTANTS.ANIMATION.EASE_OUT_CUBIC(progress);

      targetEdgeIds.forEach((edgeId) => {
        if (excludeEdges.includes(edgeId)) {
          return;
        }

        const edge = this.edges.get(edgeId);
        if (!edge) return;

        const defaultColor =
          edge.dashes || edge.label === "AND"
            ? "{{ CREWAI_ORANGE }}"
            : getComputedStyle(document.documentElement).getPropertyValue('--edge-or-color').trim();
        const currentOpacity = edge.opacity !== undefined ? edge.opacity : 1.0;
        const currentWidth =
          edge.width !== undefined ? edge.width : CONSTANTS.EDGE.DEFAULT_WIDTH;
        const currentShadowSize =
          edge.shadow && edge.shadow.size !== undefined
            ? edge.shadow.size
            : CONSTANTS.EDGE.DEFAULT_SHADOW_SIZE;

        const targetOpacity = 1.0;
        const targetWidth = CONSTANTS.EDGE.DEFAULT_WIDTH;
        const targetShadowSize = CONSTANTS.EDGE.DEFAULT_SHADOW_SIZE;

        const newOpacity =
          currentOpacity + (targetOpacity - currentOpacity) * eased;
        const newWidth = currentWidth + (targetWidth - currentWidth) * eased;
        const newShadowSize =
          currentShadowSize + (targetShadowSize - currentShadowSize) * eased;

        const updateData = {
          id: edge.id,
          hidden: false,
          opacity: newOpacity,
          width: newWidth,
          color: {
            color: defaultColor,
            highlight: defaultColor,
            hover: defaultColor,
            inherit: false,
          },
          shadow: {
            enabled: true,
            color: "rgba(0,0,0,0.08)",
            size: newShadowSize,
            x: 1,
            y: 1,
          },
          font: {
            color: "transparent",
            background: "transparent",
          },
          arrows: {
            to: {
              enabled: true,
              scaleFactor: 0.8,
              type: "triangle",
            },
          },
        };

        if (edge.dashes) {
          const scale = Math.sqrt(newWidth / CONSTANTS.EDGE.DEFAULT_WIDTH);
          updateData.dashes = [15 * scale, 10 * scale];
        }

        this.edges.update(updateData);
      });

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    animate();
  }

  clear() {
    const allNodesList = this.nodes.get();
    const nodeRestoreAnimStart = performance.now();
    const nodeRestoreAnimDuration = CONSTANTS.ANIMATION.DURATION;

    const animate = () => {
      const elapsed = performance.now() - nodeRestoreAnimStart;
      const progress = Math.min(elapsed / nodeRestoreAnimDuration, 1);
      const eased = CONSTANTS.ANIMATION.EASE_OUT_CUBIC(progress);

      allNodesList.forEach((node) => {
        const currentOpacity = node.opacity !== undefined ? node.opacity : 1.0;
        const targetOpacity = 1.0;
        const newOpacity =
          currentOpacity + (targetOpacity - currentOpacity) * eased;
        this.nodes.update({ id: node.id, opacity: newOpacity });
      });

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    animate();

    const edgeIds = this.edges.getIds();
    const edgeRestoreAnimStart = performance.now();
    const edgeRestoreAnimDuration = CONSTANTS.ANIMATION.DURATION;

    const animateEdges = () => {
      const elapsed = performance.now() - edgeRestoreAnimStart;
      const progress = Math.min(elapsed / edgeRestoreAnimDuration, 1);
      const eased = CONSTANTS.ANIMATION.EASE_OUT_CUBIC(progress);

      edgeIds.forEach((edgeId) => {
        if (this.activeDrawerEdges.includes(edgeId)) {
          return;
        }

        const edge = this.edges.get(edgeId);
        if (!edge) return;

        const defaultColor =
          edge.dashes || edge.label === "AND"
            ? "{{ CREWAI_ORANGE }}"
            : getComputedStyle(document.documentElement).getPropertyValue('--edge-or-color').trim();
        const currentOpacity = edge.opacity !== undefined ? edge.opacity : 1.0;
        const currentWidth =
          edge.width !== undefined ? edge.width : CONSTANTS.EDGE.DEFAULT_WIDTH;
        const currentShadowSize =
          edge.shadow && edge.shadow.size !== undefined
            ? edge.shadow.size
            : CONSTANTS.EDGE.DEFAULT_SHADOW_SIZE;

        const targetOpacity = 1.0;
        const targetWidth = CONSTANTS.EDGE.DEFAULT_WIDTH;
        const targetShadowSize = CONSTANTS.EDGE.DEFAULT_SHADOW_SIZE;

        const newOpacity =
          currentOpacity + (targetOpacity - currentOpacity) * eased;
        const newWidth = currentWidth + (targetWidth - currentWidth) * eased;
        const newShadowSize =
          currentShadowSize + (targetShadowSize - currentShadowSize) * eased;

        const updateData = {
          id: edge.id,
          hidden: false,
          opacity: newOpacity,
          width: newWidth,
          color: {
            color: defaultColor,
            highlight: defaultColor,
            hover: defaultColor,
            inherit: false,
          },
          shadow: {
            enabled: true,
            color: "rgba(0,0,0,0.08)",
            size: newShadowSize,
            x: 1,
            y: 1,
          },
          font: {
            color: "transparent",
            background: "transparent",
          },
          arrows: {
            to: {
              enabled: true,
              scaleFactor: 0.8,
              type: "triangle",
            },
          },
        };

        if (edge.dashes) {
          const scale = Math.sqrt(newWidth / CONSTANTS.EDGE.DEFAULT_WIDTH);
          updateData.dashes = [15 * scale, 10 * scale];
        }

        this.edges.update(updateData);
      });

      if (progress < 1) {
        requestAnimationFrame(animateEdges);
      }
    };

    animateEdges();

    this.highlightedNodes = [];
    this.highlightedEdges = [];

    this.canvas.style.transition = `opacity ${CONSTANTS.ANIMATION.DURATION}ms ease-out`;
    this.canvas.style.opacity = "0";
    setTimeout(() => {
      this.canvas.classList.remove("visible");
      this.canvas.style.opacity = "1";
      this.canvas.style.transition = "";
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.network.redraw();
    }, CONSTANTS.ANIMATION.DURATION);
  }
}

class DrawerManager {
  constructor(network, nodes, edges, animationManager, triggeredByHighlighter, networkManager) {
    this.network = network;
    this.nodes = nodes;
    this.edges = edges;
    this.animationManager = animationManager;
    this.triggeredByHighlighter = triggeredByHighlighter;
    this.networkManager = networkManager;

    this.elements = {
      drawer: document.getElementById("drawer"),
      overlay: document.getElementById("drawer-overlay"),
      title: document.getElementById("drawer-node-name"),
      content: document.getElementById("drawer-content"),
      openIdeButton: document.getElementById("drawer-open-ide"),
      closeButton: document.getElementById("drawer-close"),
      navControls: document.querySelector(".nav-controls"),
      legendPanel: document.getElementById("legend-panel"),
    };

    this.activeNodeId = null;
    this.activeEdges = [];

    this.setupEventListeners();
  }

  setupEventListeners() {
    this.elements.overlay.addEventListener("click", () => this.close());
    this.elements.closeButton.addEventListener("click", () => this.close());

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        this.close();
      }
    });
  }

  open(nodeName, metadata) {
    this.elements.title.textContent = nodeName;
    this.setupIdeButton(metadata);
    this.renderContent(nodeName, metadata);
    this.animateOpen();
  }

  setupIdeButton(metadata) {
    if (metadata.source_file && metadata.source_start_line) {
      this.elements.openIdeButton.style.display = "flex";
      this.elements.openIdeButton.onclick = () => this.openInIDE(metadata);
    } else {
      this.elements.openIdeButton.style.display = "none";
    }
  }

  openInIDE(metadata) {
    const filePath = metadata.source_file;
    const lineNum = metadata.source_start_line;
    const detectedIDE = this.detectIDE();

    const ideUrls = {
      pycharm: `pycharm://open?file=${filePath}&line=${lineNum}`,
      vscode: `vscode://file/${filePath}:${lineNum}`,
      jetbrains: `jetbrains://open?file=${encodeURIComponent(filePath)}&line=${lineNum}`,
      auto: `pycharm://open?file=${filePath}&line=${lineNum}`,
    };

    const ideUrl = ideUrls[detectedIDE] || ideUrls.auto;

    const link = document.createElement("a");
    link.href = ideUrl;
    link.target = "_blank";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    const fallbackText = `${filePath}:${lineNum}`;
    navigator.clipboard.writeText(fallbackText).catch(() => {});
  }

  detectIDE() {
    const savedIDE = localStorage.getItem("preferred_ide");
    if (savedIDE) return savedIDE;
    if (navigator.userAgent.includes("JetBrains")) return "jetbrains";
    return "auto";
  }

  renderContent(nodeName, metadata) {
    let content = "";

    content += this.renderMetadata(metadata);

    if (metadata.source_code) {
      content += this.renderSourceCode(metadata);
    }

    this.elements.content.innerHTML = content;
    this.attachContentEventListeners(nodeName);

    // Initialize Lucide icons in the newly rendered drawer content
    if (typeof lucide !== 'undefined') {
      lucide.createIcons();
    }
  }

  renderTriggerCondition(metadata) {
    if (metadata.trigger_condition) {
      return this.renderConditionTree(metadata.trigger_condition);
    } else if (metadata.trigger_methods) {
      const uniqueTriggers = [...new Set(metadata.trigger_methods)];
      const grouped = this.groupByIdenticalAction(uniqueTriggers);

      return `
        <ul class="drawer-list">
          ${grouped.map((group) => {
            if (group.items.length === 1) {
              return `<li><span class="drawer-code-link" data-node-id="${group.items[0]}">${group.items[0]}</span></li>`;
            } else {
              const groupId = group.items.join(',');
              return `
                <li>
                  <div class="trigger-group" data-trigger-items="${groupId}" style="border-left: 2px solid var(--text-secondary); padding: 4px 0 4px 8px; margin: 2px 0; border-radius: 3px; cursor: pointer; transition: background 0.15s ease;">
                    <div class="trigger-group-label" style="font-size: 11px; font-weight: 600; color: var(--text-secondary); margin-bottom: 4px; user-select: none;">
                      ${group.items.length} routes <span style="opacity: 0.5; font-size: 9px;">→</span>
                    </div>
                    <div class="trigger-group-items" style="display: flex; flex-wrap: wrap; gap: 4px; pointer-events: none;">
                      ${group.items.map((t) => `<span class="drawer-code" style="opacity: 0.7;">${t}</span>`).join("")}
                    </div>
                  </div>
                </li>
              `;
            }
          }).join("")}
        </ul>
      `;
    }
    return "";
  }

  groupByIdenticalAction(triggerIds) {
    const nodeData = '{{ nodeData }}';
    const allEdges = this.edges.get();
    const activeNodeId = this.activeNodeId;

    const triggerPaths = new Map();

    triggerIds.forEach(triggerId => {
      const pathSignature = this.getPathSignature(triggerId, activeNodeId, allEdges, nodeData);
      if (!triggerPaths.has(pathSignature)) {
        triggerPaths.set(pathSignature, []);
      }
      triggerPaths.get(pathSignature).push(triggerId);
    });

    return Array.from(triggerPaths.values()).map(items => ({ items }));
  }

  getPathSignature(triggerNodeId, activeNodeId, allEdges, nodeData) {
    const connectingEdges = [];
    const direct = allEdges.filter(
      (edge) => edge.from === triggerNodeId && edge.to === activeNodeId
    );
    if (direct.length > 0) {
      return direct.map(e => e.id).sort().join(',');
    }

    const activeMetadata = nodeData[activeNodeId];
    if (activeMetadata && activeMetadata.trigger_methods && activeMetadata.trigger_methods.includes(triggerNodeId)) {
      for (const [nodeName, nodeInfo] of Object.entries(nodeData)) {
        if (nodeInfo.router_paths && nodeInfo.router_paths.includes(triggerNodeId)) {
          const routerEdges = allEdges.filter(
            (edge) => edge.from === nodeName && edge.dashes
          );

          const matchingEdge = routerEdges.find(edge => edge.label === triggerNodeId);
          if (matchingEdge) {
            if (matchingEdge.to === activeNodeId) {
              return matchingEdge.id;
            }

            const pathToActive = allEdges.filter(
              (edge) => edge.from === matchingEdge.to && edge.to === activeNodeId
            );

            if (pathToActive.length > 0) {
              return [matchingEdge.id, ...pathToActive.map(e => e.id)].sort().join(',');
            }
          }

          for (const routerEdge of routerEdges) {
            if (routerEdge.to === activeNodeId) {
              return routerEdge.id;
            }
          }
        }
      }
    }

    return triggerNodeId;
  }

  renderConditionTree(condition, depth = 0) {
    if (typeof condition === "string") {
      return `<span class="drawer-code-link trigger-leaf" data-node-id="${condition}">${condition}</span>`;
    }

    if (condition.type === "AND" || condition.type === "OR") {
      const conditionType = condition.type;
      const color = conditionType === "AND" ? "{{ CREWAI_ORANGE }}" : "var(--text-secondary)";
      const bgColor = conditionType === "AND" ? "rgba(255,90,80,0.08)" : "rgba(102,102,102,0.06)";
      const hoverBg = conditionType === "AND" ? "rgba(255,90,80,0.15)" : "rgba(102,102,102,0.12)";

      const triggerIds = this.extractTriggerIds(condition);
      const triggerIdsJson = JSON.stringify(triggerIds).replace(/"/g, '&quot;');

      const stringChildren = condition.conditions.filter(c => typeof c === 'string');
      const nonStringChildren = condition.conditions.filter(c => typeof c !== 'string');

      let children = "";

      if (nonStringChildren.length > 0) {
        children += nonStringChildren.map(sub => this.renderConditionTree(sub, depth + 1)).join("");
      }

      if (stringChildren.length > 0) {
        const grouped = this.groupByIdenticalAction(stringChildren);
        children += grouped.map((group) => {
          if (group.items.length === 1) {
            return this.renderConditionTree(group.items[0], depth + 1);
          } else {
            const groupId = group.items.join(',');
            const groupColor = conditionType === "AND" ? "{{ CREWAI_ORANGE }}" : "var(--text-secondary)";
            const groupBgColor = conditionType === "AND" ? "rgba(255,90,80,0.08)" : "rgba(102,102,102,0.06)";
            const groupHoverBg = conditionType === "AND" ? "rgba(255,90,80,0.15)" : "rgba(102,102,102,0.12)";
            return `
              <div class="trigger-group" data-trigger-items="${groupId}" style="border-left: 2px solid ${groupColor}; padding: 8px 0 8px 12px; margin: 4px 0; border-radius: 4px; cursor: pointer; transition: background 0.2s ease;" onmouseover="this.style.background='${groupHoverBg}'" onmouseout="this.style.background='transparent'">
                <div class="trigger-group-label" style="color: ${groupColor}; font-size: 11px; font-weight: 600; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; background: ${groupBgColor}; padding: 3px 8px; border-radius: 3px; display: inline-flex; align-items: center; gap: 4px; user-select: none;">
                  ${group.items.length} routes <i data-lucide="chevron-down" style="width: 14px; height: 14px; color: ${groupColor};"></i>
                </div>
                <div class="trigger-group-items" style="display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; pointer-events: none;">
                  ${group.items.map((t) => `<span class="drawer-code-link trigger-leaf" style="opacity: 0.7; cursor: default;">${t}</span>`).join("")}
                </div>
              </div>
            `;
          }
        }).join("");
      }

      return `
        <div class="condition-group" data-trigger-group="${triggerIdsJson}" style="border-left: 2px solid ${color}; padding: 8px 0 8px 12px; margin: 4px 0; transition: background 0.2s ease; position: relative; border-radius: 4px;" onmouseover="this.style.background='${hoverBg}'" onmouseout="this.style.background='transparent'">
          <div class="condition-label" data-condition-label="${conditionType}" style="color: ${color}; font-size: 11px; font-weight: 600; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; background: ${bgColor}; padding: 3px 8px; border-radius: 3px; display: inline-flex; align-items: center; gap: 4px; cursor: pointer; user-select: none;">
            ${conditionType} <i data-lucide="chevron-down" style="width: 14px; height: 14px; color: ${color};"></i>
          </div>
          <div class="condition-children" style="margin-top: 4px;">
            ${children}
          </div>
        </div>
      `;
    }

    return "";
  }

  extractTriggerIds(condition) {
    if (typeof condition === "string") {
      return [condition];
    }

    if (condition.type === "AND" || condition.type === "OR") {
      const ids = [];
      condition.conditions.forEach(sub => {
        ids.push(...this.extractTriggerIds(sub));
      });
      return ids;
    }

    return [];
  }

  renderMetadata(metadata) {
    console.log('renderMetadata called with:', metadata);
    let metadataContent = "";

    const nodeType = metadata.type || "unknown";
    const typeBadgeColor =
      nodeType === "start" || nodeType === "router"
        ? "{{ CREWAI_ORANGE }}"
        : "{{ DARK_GRAY }}";
    metadataContent += `
            <div class="drawer-section">
                <div class="drawer-section-title">Type</div>
                <span class="drawer-badge" style="background: ${typeBadgeColor}; color: white;">${nodeType}</span>
            </div>
        `;

    if (metadata.condition_type) {
      const conditionColors = {
        AND: { color: "{{ CREWAI_ORANGE }}", bg: "rgba(255,90,80,0.12)" },
        IF: { color: "{{ CREWAI_ORANGE }}", bg: "rgba(255,90,80,0.18)" },
        default: { color: "{{ GRAY }}", bg: "rgba(102,102,102,0.12)" },
      };

      const { color, bg } =
        conditionColors[metadata.condition_type] || conditionColors.default;

      metadataContent += `
                <div class="drawer-section">
                    <div class="drawer-section-title">Condition</div>
                    <span class="drawer-badge" style="background: ${bg}; color: ${color};">${metadata.condition_type}</span>
                </div>
            `;
    }

    console.log('Checking trigger data:', {
      has_trigger_condition: !!metadata.trigger_condition,
      has_trigger_methods: !!(metadata.trigger_methods && metadata.trigger_methods.length > 0)
    });

    if (metadata.trigger_condition || (metadata.trigger_methods && metadata.trigger_methods.length > 0)) {
      console.log('Rendering Triggered By section');
      metadataContent += `
                <div class="drawer-section">
                    <div class="drawer-section-title">Triggered By</div>
                    ${this.renderTriggerCondition(metadata)}
                </div>
            `;
    }

    if (metadata.router_paths && metadata.router_paths.length > 0) {
      const uniqueRouterPaths = [...new Set(metadata.router_paths)];
      const routerPathsJson = JSON.stringify(uniqueRouterPaths).replace(/"/g, '&quot;');
      metadataContent += `
                <div class="drawer-section">
                    <div class="drawer-section-title router-paths-title" data-router-paths="${routerPathsJson}" style="cursor: pointer; display: inline-flex; align-items: center; gap: 4px;">
                        Router Paths <i data-lucide="chevron-down" style="width: 14px; height: 14px; color: var(--text-primary);"></i>
                    </div>
                    <ul class="drawer-list">
                        ${uniqueRouterPaths.map((p) => `<li><span class="drawer-code-link" data-node-id="${p}" style="color: {{ CREWAI_ORANGE }}; border-color: rgba(255,90,80,0.3);">${p}</span></li>`).join("")}
                    </ul>
                </div>
            `;
    }

    return metadataContent
      ? `<div class="drawer-metadata-grid">${metadataContent}</div>`
      : "";
  }

  renderSourceCode(metadata) {
    let lines = metadata.source_lines || metadata.source_code.split("\n");
    if (metadata.source_lines) {
      lines = lines.map((line) => line.replace(/\n$/, ""));
    }

    let minIndent = Infinity;
    lines.forEach((line) => {
      if (line.trim().length > 0) {
        const match = line.match(/^\s*/);
        const indent = match ? match[0].length : 0;
        minIndent = Math.min(minIndent, indent);
      }
    });

    const dedentedLines = lines.map((line) => {
      return line.trim().length === 0 ? "" : line.substring(minIndent);
    });

    const startLine = metadata.source_start_line || 1;
    const codeToHighlight = dedentedLines.join("\n").trim();
    const highlightedCode = highlightPython(codeToHighlight);

    const highlightedLines = highlightedCode.split("\n");
    const numberedLines = highlightedLines
      .map((line, index) => {
        const lineNum = startLine + index;
        return `<div class="code-line"><span class="line-number">${lineNum}</span>${line}</div>`;
      })
      .join("");

    let classSection = "";
    if (metadata.class_signature) {
      const highlightedClass = highlightPython(metadata.class_signature);
      let highlightedClassSignature = highlightedClass;

      if (metadata.class_line_number) {
        highlightedClassSignature = `<span class="line-number">${metadata.class_line_number}</span>${highlightedClass}`;
      }

      classSection = `
                <div>
                    <div class="accordion-subheader accordion-title">Class</div>
                    <div class="drawer-section">
                        <div class="code-block-container">
                            <pre class="code-block language-python">${highlightedClassSignature}</pre>
                        </div>
                    </div>
                </div>
            `;
    }

    return `
            <div class="accordion-section expanded">
                <div class="accordion-header">
                    <div class="accordion-title">Source Code</div>
                    <svg class="accordion-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="4,6 8,10 12,6" stroke-linecap="round" stroke-linejoin="round" />
                    </svg>
                </div>
                <div class="accordion-content">
                    <div class="drawer-section">
                        <div class="code-block-container">
                            <button class="code-copy-button" data-code="${codeToHighlight.replace(/"/g, "&quot;")}">
                                <svg class="copy-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
                                    <rect x="5" y="5" width="9" height="9" rx="1.5" />
                                    <path d="M3 11V3a2 2 0 0 1 2-2h8" />
                                </svg>
                                <svg class="check-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                                    <polyline points="3,8 6,11 13,4" stroke-linecap="round" stroke-linejoin="round" />
                                </svg>
                            </button>
                            <pre class="code-block language-python">${numberedLines}</pre>
                        </div>
                    </div>
                    ${classSection}
                </div>
            </div>
        `;
  }

  attachContentEventListeners(nodeName) {
    const copyButtons =
      this.elements.content.querySelectorAll(".code-copy-button");
    copyButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const codeText = button.getAttribute("data-code");
        navigator.clipboard
          .writeText(codeText)
          .then(() => {
            button.classList.add("copied");
            setTimeout(() => button.classList.remove("copied"), 2000);
          })
          .catch((err) => console.error("Failed to copy:", err));
      });
    });

    const accordionHeaders =
      this.elements.content.querySelectorAll(".accordion-header");
    accordionHeaders.forEach((header) => {
      header.addEventListener("click", () => {
        const accordionSection = header.closest(".accordion-section");
        accordionSection.classList.toggle("expanded");
      });
    });

    const triggeredByLinks = this.elements.content.querySelectorAll(
      ".drawer-code-link[data-node-id]",
    );
    triggeredByLinks.forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const triggerNodeId = link.getAttribute("data-node-id");
        this.triggeredByHighlighter.highlightTriggeredBy(triggerNodeId);
      });
    });

    const triggerGroups = this.elements.content.querySelectorAll(
      ".trigger-group[data-trigger-items]",
    );
    triggerGroups.forEach((group) => {
      group.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        const triggerItems = group.getAttribute("data-trigger-items").split(',');
        this.triggeredByHighlighter.highlightTriggeredByGroup(triggerItems);
      });
    });

    const conditionGroups = this.elements.content.querySelectorAll(
      ".condition-group[data-trigger-group]",
    );
    conditionGroups.forEach((group) => {
      group.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const triggerIds = JSON.parse(group.getAttribute("data-trigger-group"));
        this.triggeredByHighlighter.highlightTriggeredByGroup(triggerIds);
      });
    });

    const routerPathsTitle = this.elements.content.querySelector(
      ".router-paths-title[data-router-paths]",
    );
    if (routerPathsTitle) {
      routerPathsTitle.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.triggeredByHighlighter.highlightAllRouterPaths();
      });
    }
  }

  animateOpen() {
    this.elements.drawer.style.visibility = "visible";
    const wasAlreadyOpen = this.elements.drawer.classList.contains("open");

    if (!wasAlreadyOpen) {
      // Save current position and scale before opening drawer
      const currentPosition = this.networkManager.network.getViewPosition();
      const currentScale = this.networkManager.network.getScale();
      this.networkManager.positionBeforeDrawer = {
        position: currentPosition,
        scale: currentScale
      };

      const targetPosition = this.networkManager.calculateNetworkPosition(true);
      this.elements.drawer.classList.add("open");
      this.elements.overlay.classList.add("visible");
      this.elements.navControls.classList.add("drawer-open");
      this.elements.legendPanel.classList.add("drawer-open");
      this.networkManager.animateToPosition(targetPosition);
    } else {
      this.elements.drawer.classList.add("open");
      this.elements.overlay.classList.add("visible");
      this.elements.navControls.classList.add("drawer-open");
      this.elements.legendPanel.classList.add("drawer-open");
    }
  }

  close() {
    // Animate accordions closed before removing classes
    const accordions = this.elements.drawer.querySelectorAll(".accordion-section.expanded");
    accordions.forEach(accordion => {
      const content = accordion.querySelector(".accordion-content");
      if (content) {
        // Set explicit height for smooth animation
        content.style.height = content.scrollHeight + "px";
        // Force reflow
        content.offsetHeight;
        // Trigger collapse animation
        content.style.height = "0px";
      }
      // Remove expanded class after animation
      setTimeout(() => {
        accordion.classList.remove("expanded");
        if (content) {
          content.style.height = "";
        }
      }, CONSTANTS.ANIMATION.DURATION);
    });

    this.elements.drawer.classList.remove("open");
    this.elements.overlay.classList.remove("visible");
    this.elements.navControls.classList.remove("drawer-open");
    this.elements.legendPanel.classList.remove("drawer-open");

    if (this.activeNodeId) {
      this.activeEdges.forEach((edgeId) => {
        this.animationManager.animateEdgeWidth(
          this.edges,
          edgeId,
          CONSTANTS.EDGE.DEFAULT_WIDTH,
          CONSTANTS.ANIMATION.DURATION,
        );
      });
      this.activeNodeId = null;
      this.activeEdges = [];
    }

    this.triggeredByHighlighter.clear();
    this.elements.drawer.offsetHeight;

    // Restore the position before the drawer was opened
    if (this.networkManager.positionBeforeDrawer) {
      this.networkManager.animateToPosition(this.networkManager.positionBeforeDrawer);
      this.networkManager.positionBeforeDrawer = null;
    } else {
      this.networkManager.fitToAvailableSpace();
    }

    setTimeout(() => {
      if (!this.elements.drawer.classList.contains("open")) {
        this.elements.drawer.style.visibility = "hidden";
      }
    }, CONSTANTS.ANIMATION.DURATION);
  }

  setActiveNode(nodeId, connectedEdges) {
    this.activeNodeId = nodeId;
    this.activeEdges = connectedEdges;
  }
}

// ============================================================================
// Network Manager
// ============================================================================

class NetworkManager {
  constructor() {
    this.network = null;
    this.nodes = null;
    this.edges = null;
    this.animationManager = new AnimationManager();
    this.drawerManager = null;
    this.triggeredByHighlighter = null;

    this.hoveredNodeId = null;
    this.pressedNodeId = null;
    this.pressedEdges = [];
    this.isClicking = false;
    this.positionBeforeDrawer = null;
  }

  async initialize() {
    try {
      await loadVisCDN();

      this.nodes = new vis.DataSet('{{ nodes_list_json }}');
      this.edges = new vis.DataSet('{{ edges_list_json }}');

      const container = document.getElementById("network");
      const options = this.createNetworkOptions();

      this.network = new vis.Network(
        container,
        { nodes: this.nodes, edges: this.edges },
        options,
      );

      this.triggeredByHighlighter = new TriggeredByHighlighter(
        this.network,
        this.nodes,
        this.edges,
        document.getElementById("highlight-canvas"),
      );

      this.drawerManager = new DrawerManager(
        this.network,
        this.nodes,
        this.edges,
        this.animationManager,
        this.triggeredByHighlighter,
        this,
      );

      this.setupEventListeners();
      this.setupControls();
      this.setupTheme();

      this.network.once("stabilizationIterationsDone", () => {
        this.fitToAvailableSpace(true);
      });
    } catch (error) {
      console.error("Failed to initialize network:", error);
    }
  }

  createNetworkOptions() {
    this.nodeRenderer = new NodeRenderer(this.nodes, this);
    const nodesArray = this.nodes.get();
    const hasExplicitPositions = nodesArray.some(node =>
      node.x !== undefined && node.y !== undefined
    );

    return {
      nodes: {
        shape: "custom",
        shadow: false,
        chosen: false,
        size: 30,
        ctxRenderer: (params) => this.nodeRenderer.render(params),
        scaling: {
          min: 1,
          max: 100,
        },
      },
      edges: {
        width: CONSTANTS.EDGE.DEFAULT_WIDTH,
        hoverWidth: 0,
        labelHighlightBold: false,
        shadow: false,
        smooth: {
          enabled: true,
          type: "cubicBezier",
          roundness: 0.35,
          forceDirection: 'vertical',
        },
        font: {
          size: 13,
          align: "middle",
          color: "transparent",
          face: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          strokeWidth: 0,
          background: "transparent",
          vadjust: 0,
        },
        arrows: {
          to: {
            enabled: true,
            scaleFactor: 0.8,
            type: "triangle",
          },
        },
        arrowStrikethrough: true,
        chosen: {
          edge: false,
          label: false,
        },
      },
      physics: {
        enabled: false,
        hierarchicalRepulsion: {
          nodeDistance: CONSTANTS.NETWORK.NODE_DISTANCE,
          centralGravity: 0.0,
          springLength: CONSTANTS.NETWORK.SPRING_LENGTH,
          springConstant: 0.01,
          damping: 0.09,
        },
        solver: "hierarchicalRepulsion",
        stabilization: {
          enabled: false,
          iterations: CONSTANTS.NETWORK.STABILIZATION_ITERATIONS,
          updateInterval: 25,
        },
      },
      layout: {
        hierarchical: {
          enabled: !hasExplicitPositions,
          direction: "UD",
          sortMethod: "directed",
          levelSeparation: CONSTANTS.NETWORK.LEVEL_SEPARATION,
          nodeSpacing: CONSTANTS.NETWORK.NODE_SPACING,
          treeSpacing: CONSTANTS.NETWORK.TREE_SPACING,
          edgeMinimization: false,
          blockShifting: true,
          parentCentralization: true,
        },
      },
      interaction: {
        hover: true,
        hoverConnectedEdges: false,
        navigationButtons: false,
        keyboard: true,
        selectConnectedEdges: false,
        multiselect: false,
      },
    };
  }

  setupEventListeners() {
    this.network.on("hoverNode", (params) => {
      this.hoveredNodeId = params.node;
      document.body.style.cursor = "pointer";
      this.network.redraw();
    });

    this.network.on("blurNode", (params) => {
      if (this.hoveredNodeId === params.node) {
        this.hoveredNodeId = null;
      }
      document.body.style.cursor = "default";
      this.network.redraw();
    });

    this.network.on("selectNode", (params) => {
      if (params.nodes.length > 0) {
        this.pressedNodeId = params.nodes[0];
        this.pressedEdges = this.network.getConnectedEdges(this.pressedNodeId);
        this.network.redraw();
      }
    });

    this.network.on("deselectNode", () => {
      if (this.pressedNodeId) {
        setTimeout(() => {
          if (this.isClicking) {
            this.isClicking = false;
            this.pressedNodeId = null;
            this.pressedEdges = [];
            return;
          }

          this.pressedEdges.forEach((edgeId) => {
            if (!this.drawerManager.activeEdges.includes(edgeId)) {
              this.animationManager.animateEdgeWidth(
                this.edges,
                edgeId,
                CONSTANTS.EDGE.DEFAULT_WIDTH,
                150,
              );
            }
          });

          this.pressedNodeId = null;
          this.pressedEdges = [];
          this.network.redraw();
        }, 10);
      }
    });

    this.network.on("click", (params) => {
      if (params.nodes.length > 0) {
        this.pressedNodeId = params.nodes[0];
        this.network.redraw();
        this.handleNodeClick(params.nodes[0]);
      } else if (params.edges.length === 0) {
        this.pressedNodeId = null;
        this.network.redraw();
        this.drawerManager.close();
      }
    });

  }

  handleNodeClick(nodeId) {
    const node = this.nodes.get(nodeId);
    const nodeData = '{{ nodeData }}';
    const metadata = nodeData[nodeId];

    this.isClicking = true;
    if (this.drawerManager && this.drawerManager.activeNodeId === nodeId) {
      this.drawerManager.close();
      return;
    }

    const connectedEdges = this.network.getConnectedEdges(nodeId);

    const allEdges = this.edges.get();
    const connectedNodeIds = new Set([nodeId]);

    connectedEdges.forEach((edgeId) => {
      const edge = allEdges.find(e => e.id === edgeId);
      if (edge) {
        if (edge.from === nodeId) {
          connectedNodeIds.add(edge.to);
        } else if (edge.to === nodeId) {
          connectedNodeIds.add(edge.from);
        }
      }
    });
    const allNodes = this.nodes.get();
    allNodes.forEach((n) => {
      this.nodes.update({ id: n.id, opacity: 1.0 });
    });
    this.triggeredByHighlighter.highlightedNodes = [];
    this.triggeredByHighlighter.highlightedEdges = [];

    // Animate all edges back to default, excluding the ones we'll highlight
    this.triggeredByHighlighter.resetEdgesToDefault(null, connectedEdges);

    this.drawerManager.setActiveNode(nodeId, connectedEdges);
    this.triggeredByHighlighter.setActiveDrawer(nodeId, connectedEdges);

    setTimeout(() => {
      connectedEdges.forEach((edgeId) => {
        this.animationManager.animateEdgeWidth(this.edges, edgeId, 5, 200);
      });
    }, 15);

    this.drawerManager.open(nodeId, metadata);
    this.network.redraw();
  }

  setupControls() {
    const ideSelector = document.getElementById("ide-selector");
    const savedIDE = localStorage.getItem("preferred_ide") || "auto";
    ideSelector.value = savedIDE;
    ideSelector.addEventListener("change", function () {
      localStorage.setItem("preferred_ide", this.value);
    });

    document.getElementById("zoom-in").addEventListener("click", () => {
      const scale = this.network.getScale();
      this.network.moveTo({
        scale: scale * 1.2,
        animation: { duration: 200, easingFunction: "easeInOutQuad" },
      });
    });

    document.getElementById("zoom-out").addEventListener("click", () => {
      const scale = this.network.getScale();
      this.network.moveTo({
        scale: scale * 0.8,
        animation: { duration: 200, easingFunction: "easeInOutQuad" },
      });
    });

    document.getElementById("fit").addEventListener("click", () => {
      this.fitToAvailableSpace();
    });

    this.setupExportControls();
  }

  setupExportControls() {
    document.getElementById("export-png").addEventListener("click", () => {
      const script = document.createElement("script");
      script.src =
        "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js";
      script.onload = () => {
        html2canvas(document.getElementById("network-container")).then(
          (canvas) => {
            const link = document.createElement("a");
            link.download = "flow.png";
            link.href = canvas.toDataURL();
            link.click();
          },
        );
      };
      document.head.appendChild(script);
    });

    document.getElementById("export-pdf").addEventListener("click", () => {
      const script1 = document.createElement("script");
      script1.src =
        "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js";
      script1.onload = () => {
        const script2 = document.createElement("script");
        script2.src =
          "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";
        script2.onload = () => {
          html2canvas(document.getElementById("network-container")).then(
            (canvas) => {
              const imgData = canvas.toDataURL("image/png");
              const { jsPDF } = window.jspdf;
              const pdf = new jsPDF({
                orientation:
                  canvas.width > canvas.height ? "landscape" : "portrait",
                unit: "px",
                format: [canvas.width, canvas.height],
              });
              pdf.addImage(imgData, "PNG", 0, 0, canvas.width, canvas.height);
              pdf.save("flow.pdf");
            },
          );
        };
        document.head.appendChild(script2);
      };
      document.head.appendChild(script1);
    });

    // document.getElementById("export-json").addEventListener("click", () => {
    //   const dagData = '{{ dagData }}';
    //   const dataStr = JSON.stringify(dagData, null, 2);
    //   const blob = new Blob([dataStr], { type: "application/json" });
    //   const url = URL.createObjectURL(blob);
    //   const link = document.createElement("a");
    //   link.download = "flow_dag.json";
    //   link.href = url;
    //   link.click();
    //   URL.revokeObjectURL(url);
    // });
  }

  calculateNetworkPosition(isDrawerOpen, centerScreen = false) {
    const infoBox = document.getElementById("info");
    const infoRect = infoBox.getBoundingClientRect();
    const leftEdge = infoRect.right + 40; // 40px padding after legend
    const rightEdge = isDrawerOpen ? window.innerWidth - CONSTANTS.DRAWER.WIDTH - 40 : window.innerWidth - 40;
    const availableWidth = rightEdge - leftEdge;

    // Use true screen center for initial position, otherwise use available space center
    const canvas = this.network ? this.network.canvas.frame.canvas : document.getElementById("network");
    const canvasRect = canvas.getBoundingClientRect();
    const domCenterX = centerScreen ? canvasRect.left + canvas.clientWidth / 2 : leftEdge + (availableWidth / 2);

    const nodePositions = this.network.getPositions();
    const nodeIds = Object.keys(nodePositions);

    if (nodeIds.length === 0) return null;
    const canvasWidth = canvas.clientWidth;
    const canvasHeight = canvas.clientHeight;

    const padding = 30;
    const maxNodeWidth = 200;
    const maxNodeHeight = 60;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    nodeIds.forEach(id => {
      const pos = nodePositions[id];
      minX = Math.min(minX, pos.x - maxNodeWidth / 2);
      maxX = Math.max(maxX, pos.x + maxNodeWidth / 2);
      minY = Math.min(minY, pos.y - maxNodeHeight / 2);
      maxY = Math.max(maxY, pos.y + maxNodeHeight / 2);
    });

    const networkWidth = maxX - minX;
    const networkHeight = maxY - minY;
    const networkCenterX = (minX + maxX) / 2;
    const networkCenterY = (minY + maxY) / 2;
    const scaleX = availableWidth / (networkWidth + padding * 2);
    const scaleY = (canvasHeight - padding * 2) / (networkHeight + padding * 2);
    const scale = Math.min(scaleX, scaleY);
    const targetDOMX = domCenterX;
    const canvasCenterDOMX = canvasRect.left + canvasWidth / 2;
    const domShift = targetDOMX - canvasCenterDOMX;
    const networkShift = domShift / scale;

    return {
      position: {
        x: networkCenterX - networkShift,
        y: networkCenterY,
      },
      scale: scale,
    };
  }

  animateToPosition(targetPosition) {
    if (!targetPosition) return;

    this.network.moveTo({
      position: targetPosition.position,
      scale: targetPosition.scale,
      animation: {
        duration: 300,
        easingFunction: "easeInOutCubic"
      },
    });
  }

  fitToAvailableSpace(centerScreen = false) {
    const drawer = document.getElementById("drawer");
    const isDrawerOpen = drawer.classList.contains("open");
    const targetPosition = this.calculateNetworkPosition(isDrawerOpen, centerScreen);
    this.animateToPosition(targetPosition);
  }

  setupTheme() {
    const themeToggle = document.getElementById("theme-toggle");
    const htmlElement = document.documentElement;

    const updateEdgeColors = () => {
      const orEdgeColor = getComputedStyle(document.documentElement).getPropertyValue('--edge-or-color').trim();

      this.edges.forEach((edge) => {
        let edgeColor;
        if (edge.dashes || edge.label === "AND") {
          edgeColor = "{{ CREWAI_ORANGE }}";
        } else {
          edgeColor = orEdgeColor;
        }

        const updateData = {
          id: edge.id,
          color: {
            color: edgeColor,
            highlight: edgeColor,
          },
          font: {
            color: "transparent",
            background: "transparent",
          },
        };

        this.edges.update(updateData);
      });

      this.network.redraw();
    };

    const updateThemeIcon = (isDark) => {
      const iconName = isDark ? 'sun' : 'moon';
      themeToggle.title = isDark ? "Toggle Light Mode" : "Toggle Dark Mode";

      // Replace the icon HTML entirely and reinitialize Lucide
      themeToggle.innerHTML = `<i data-lucide="${iconName}" style="width: 18px; height: 18px;"></i>`;

      // Reinitialize Lucide icons for the specific button
      if (typeof lucide !== 'undefined') {
        lucide.createIcons({
          elements: themeToggle.querySelectorAll('[data-lucide]')
        });
      }
    };

    // Set up click handler FIRST before any icon updates
    themeToggle.addEventListener("click", () => {
      const currentTheme = htmlElement.getAttribute("data-theme");
      const newTheme = currentTheme === "dark" ? "light" : "dark";

      if (newTheme === "dark") {
        htmlElement.setAttribute("data-theme", "dark");
        updateThemeIcon(true);
      } else {
        htmlElement.removeAttribute("data-theme");
        updateThemeIcon(false);
      }

      localStorage.setItem("theme", newTheme);

      // Clear color cache to ensure theme-dependent colors are recalculated
      if (this.nodeRenderer) {
        this.nodeRenderer.colorCache.clear();
      }

      // Update edge colors and redraw network with new theme
      setTimeout(updateEdgeColors, 50);
    });

    // Initialize theme after click handler is set up
    const savedTheme = localStorage.getItem("theme") || "dark";
    if (savedTheme === "dark") {
      htmlElement.setAttribute("data-theme", "dark");
      updateThemeIcon(true);
      setTimeout(updateEdgeColors, 0);
    } else {
      updateThemeIcon(false);
      setTimeout(updateEdgeColors, 0);
    }
  }
}

// ============================================================================
// Application Entry Point
// ============================================================================

(async () => {
  // Initialize Lucide icons first (before theme setup)
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }

  const networkManager = new NetworkManager();
  await networkManager.initialize();

  // Re-initialize Lucide icons after theme is set up
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }
})();
