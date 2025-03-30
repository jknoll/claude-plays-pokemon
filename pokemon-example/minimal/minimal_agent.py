#!/usr/bin/env python3
# /// script
# dependencies = [
#   "morphcloud",
#   "requests",
#   "pillow",
#   "rich",
#   "anthropic",
#   "opencv-python",
#   "numpy",
# ]
# ///


"""
Run a non-interactive server agent that plays Pokemon automatically.
This script combines the EmulatorClient, ServerAgent, and runner into a single file.
"""
import argparse
import base64
import copy
import io
import json
import logging
import os
import requests
import sys
from PIL import Image
from anthropic import Anthropic
from morphcloud.api import MorphCloudClient
import cv2
import numpy as np

# Set up logging - this will be configured properly in main() based on command line args
logger = logging.getLogger(__name__)

# Configuration
MAX_TOKENS = 4096
MODEL_NAME = "claude-3-7-sonnet-20250219"
TEMPERATURE = 0.7
USE_NAVIGATOR = True


class EmulatorClient:
    def __init__(self, host='127.0.0.1', port=9876):
        # Check if host already includes the protocol, if not add http://
        if host.startswith('http://') or host.startswith('https://'):
            # For MorphVM URLs, don't append port as it's handled by the URL routing
            if "cloud.morph.so" in host or port is None:
                self.base_url = host
            # For other URLs, handle port as before
            elif ":" not in host.split('/')[-1]:
                self.base_url = f"{host}:{port}"
            else:
                # Host already has port, use it as is
                self.base_url = host
        else:
            # For MorphVM URLs, don't append port
            if "cloud.morph.so" in host:
                self.base_url = f"https://{host}"
            else:
                self.base_url = f"http://{host}:{port}"
        logger.info(f"Initialized client connecting to {self.base_url}")
        
    def get_screenshot(self):
        """Get current screenshot as PIL Image"""
        response = requests.get(f"{self.base_url}/api/screenshot")
        if response.status_code != 200:
            logger.error(f"Error getting screenshot: {response.status_code}")
            return None
        return Image.open(io.BytesIO(response.content))
    
    def get_screenshot_base64(self):
        """Get current screenshot as base64 string"""
        response = requests.get(f"{self.base_url}/api/screenshot")
        if response.status_code != 200:
            logger.error(f"Error getting screenshot: {response.status_code}")
            return ""
        return base64.b64encode(response.content).decode('utf-8')
    
    def get_game_state(self):
        """Get complete game state from server"""
        response = requests.get(f"{self.base_url}/api/game_state")
        if response.status_code != 200:
            logger.error(f"Error response from server: {response.status_code} - {response.text}")
            return {}
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Response content: {response.text[:100]}...")
            return {}
    
    # Compatibility methods to match Emulator interface
    def get_state_from_memory(self):
        """Get game state string - mimics Emulator.get_state_from_memory()"""
        state_data = self.get_game_state()
        return state_data.get('game_state', '')
    
    def get_collision_map(self):
        """Get collision map - mimics Emulator.get_collision_map()"""
        state_data = self.get_game_state()
        return state_data.get('collision_map', '')
    
    def get_valid_moves(self):
        """Get valid moves - mimics Emulator.get_valid_moves()"""
        state_data = self.get_game_state()
        return state_data.get('valid_moves', [])
    
    def find_path(self, row, col):
        """Find path to position - mimics Emulator.find_path()"""
        result = self.navigate(row, col)
        if not isinstance(result, dict):
            return "Failed to navigate", []
        return result.get('status', 'Navigation failed'), result.get('path', [])
    
    def press_buttons(self, buttons, wait=True, include_state=False, include_screenshot=False):
        """Press a sequence of buttons on the Game Boy
        
        Args:
            buttons: List of buttons to press
            wait: Whether to pause briefly after each button press
            include_state: Whether to include game state in response
            include_screenshot: Whether to include screenshot in response
            
        Returns:
            dict: Response data which may include button press result, game state, and screenshot
        """
        data = {
            "buttons": buttons,
            "wait": wait,
            "include_state": include_state,
            "include_screenshot": include_screenshot
        }
        response = requests.post(f"{self.base_url}/api/press_buttons", json=data)
        if response.status_code != 200:
            logger.error(f"Error pressing buttons: {response.status_code} - {response.text}")
            return {"error": f"Error: {response.status_code}"}
        
        return response.json()
    
    def navigate(self, row, col, include_state=False, include_screenshot=False):
        """Navigate to a specific position on the grid
        
        Args:
            row: Target row coordinate
            col: Target column coordinate
            include_state: Whether to include game state in response
            include_screenshot: Whether to include screenshot in response
            
        Returns:
            dict: Response data which may include navigation result, game state, and screenshot
        """
        data = {
            "row": row,
            "col": col,
            "include_state": include_state,
            "include_screenshot": include_screenshot
        }
        response = requests.post(f"{self.base_url}/api/navigate", json=data)
        if response.status_code != 200:
            logger.error(f"Error navigating: {response.status_code} - {response.text}")
            return {"status": f"Error: {response.status_code}", "path": []}
        
        return response.json()
    
    def read_memory(self, address):
        """Read a specific memory address"""
        response = requests.get(f"{self.base_url}/api/memory/{address}")
        if response.status_code != 200:
            logger.error(f"Error reading memory: {response.status_code} - {response.text}")
            return {"error": f"Error: {response.status_code}"}
        return response.json()
    
    def load_state(self, state_path):
        """Load a saved state"""
        data = {
            "state_path": state_path
        }
        response = requests.post(f"{self.base_url}/api/load_state", json=data)
        if response.status_code != 200:
            logger.error(f"Error loading state: {response.status_code} - {response.text}")
            return {"error": f"Error: {response.status_code}"}
        return response.json()
    
    def save_screenshot(self, filename="screenshot.png"):
        """Save current screenshot to a file"""
        screenshot = self.get_screenshot()
        if screenshot:
            screenshot.save(filename)
            logger.info(f"Screenshot saved as {filename}")
            return True
        return False
    
    def initialize(self):
        """Empty initialize method for compatibility with Emulator"""
        logger.info("Client initialization requested (compatibility method)")
        # Check if server is ready
        try:
            response = requests.get(f"{self.base_url}/api/status")
            status = response.json()
            ready = status.get('ready', False)
            if ready:
                logger.info("Server reports ready status")
            else:
                logger.warning("Server reports not ready")
            return ready
        except Exception as e:
            logger.error(f"Error checking server status: {e}")
            return False
    
    def stop(self):
        """Empty stop method for compatibility with Emulator"""
        logger.info("Client stop requested (compatibility method)")
        # Nothing to do for client
        pass


def get_screenshot_base64(screenshot, upscale=1):
    """Convert PIL image to base64 string."""
    # Resize if needed
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        screenshot = screenshot.resize(new_size)

    # Convert to base64
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


class ServerAgent:
    def __init__(self, server_host='127.0.0.1', server_port=9876, max_history=60, display_config=None):
        """Initialize the server agent."""
        self.client = EmulatorClient(host=server_host, port=server_port)
        self.anthropic = Anthropic()
        self.running = True
        self.message_history = [{"role": "user", "content": "You may now begin playing."}]
        self.max_history = max_history
        self.text_map = ""
        self.current_location = ""
        self.target_location = ""
        self.navigation_instructions = ""
        
        # Set display configuration with defaults
        self.display_config = display_config or {
            'show_game_state': False,
            'show_collision_map': False,
            'quiet_mode': False
        }
        
        # SLAM map disabled
        # self.slam_map = SLAMMap(canvas_size=(5000, 5000), initial_offset=(2500, 2500))
        
        # Log initialization with chosen configuration
        logger.debug(f"Agent initialized with display config: {self.display_config}")
        
        # Check if the server is ready
        if not self.client.initialize():
            logger.error("Server not ready - please start the server before running the agent")
            raise RuntimeError("Server not ready")

    SYSTEM_PROMPT = """You are playing Pokemon Red. You can see the game screen and control the game by executing emulator commands: check your tools! for example, 'navigate_to' can help you move in the overworld. Before each action, explain your reasoning briefly, then use the available actions to control the game.

Your goal is to play through Pokemon Red and eventually defeat the Elite Four. Make decisions based on what you see on the screen.

Current Location: {current_location}
Target Location: {target_location}
Current Map Description: {text_map}

Navigation Instructions:
{navigation_instructions}

Game Progression Strategy:
1. Always work towards reaching the target location
2. Use the navigation tool to move efficiently in the overworld
3. Interact with NPCs and objects that might help progress the story

Before each action:
1. You may check if you're moving towards your target location using the navigation instructions, but these instructions are not always accurate.
2. Consider what obstacles might be in your way
3. Use the Text Map to understand your surroundings
4. Explain your reasoning, then execute your chosen commands

Think like Indiana Jones - be methodical in your exploration but always keep your ultimate goal in mind. Use your current location and the map description to make informed decisions about where to go next.

The conversation history may occasionally be summarized to save context space. If you see a message labeled "CONVERSATION HISTORY SUMMARY", this contains the key information about your progress so far. Use this information to maintain continuity in your gameplay."""

    SUMMARY_PROMPT = """I need you to create a detailed summary of our conversation history up to this point. This summary will replace the full conversation history to manage the context window.

Please include:
1. Key game events and milestones you've reached
2. Important decisions you've made
3. Current objectives or goals you're working toward
4. Your current location and Pokémon team status
5. Any strategies or plans you've mentioned

The summary should be comprehensive enough that you can continue gameplay without losing important context about what has happened so far."""
    EXPLORATION_PROMPT = """You have been the same area for a long while, and you have not been able to progress. You need to be out to explore more and interact with the environment."""
    AVAILABLE_TOOLS = [
        {
            "name": "press_buttons",
            "description": "Press a sequence of buttons on the Game Boy.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "buttons": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["a", "b", "start", "select", "up", "down", "left", "right"]
                        },
                        "description": "List of buttons to press in sequence. Valid buttons: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'"
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "Whether to wait for a brief period after pressing each button. Defaults to true."
                    }
                },
                "required": ["buttons"],
            },
        }
    ]

    # Add navigation tool if enabled
    if USE_NAVIGATOR:
        AVAILABLE_TOOLS.append({
            "name": "navigate_to",
            "description": "Automatically navigate to a position on the map grid. The screen is divided into a 9x10 grid, with the top-left corner as (0, 0). This tool is only available in the overworld.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "row": {
                        "type": "integer",
                        "description": "The row coordinate to navigate to (0-8)."
                    },
                    "col": {
                        "type": "integer",
                        "description": "The column coordinate to navigate to (0-9)."
                    }
                },
                "required": ["row", "col"],
            },
        })
    
    OBSERVATION_PROMPT = """Think Like Indiana Jones. Based solely on textual observations from a screenshot, create a succinct, updated map that highlights only significant features like structures, exploration pathways, and paths. Keep it to 100 words or less. Update the current map: {self.text_map}"""

    def update_location_from_state(self, game_state):
        """Extract current location from game state."""
        if isinstance(game_state, str):
            for line in game_state.split('\n'):
                if line.startswith("Location:"):
                    self.current_location = line.replace("Location:", "").strip()
                    logger.info(f"Current location updated: {self.current_location}")
                    break

    def get_navigation_instructions(self):
        """Ask Claude for specific navigation instructions to reach target location."""
        navigation_prompt = """Based on Pokemon Red's world map and game progression, I need specific instructions to reach the target location.
        Current location: {current_location}
        Target location: {target_location}
        Current map description: {text_map}
        
        Provide step-by-step directions to reach the target location. Include:
        1. The optimal path/route to take
        2. Any obstacles or requirements to progress (e.g., badges needed, items required)
        3. Key locations to pass through
        
        Keep it concise and clear."""

        # Create message for Claude
        response = self.anthropic.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": navigation_prompt.format(
                    current_location=self.current_location,
                    target_location=self.target_location,
                    text_map=self.text_map
                )
            }]
        )

        # Extract navigation instructions
        instructions = " ".join([block.text for block in response.content if block.type == "text"]).strip()
        
        # Log the navigation instructions
        logger.info("Navigation Instructions:")
        logger.info("-" * 50)
        logger.info(f"From: {self.current_location}")
        logger.info(f"To: {self.target_location}")
        logger.info("-" * 25)
        logger.info(instructions)
        logger.info("-" * 50)
        
        return instructions

    def update_target_location(self):
        """Ask Claude for the next target location based on game progress."""
        target_prompt = """Based on Pokemon Red's game progression, what should be my next target location to advance towards beating the game? Consider:
        1. Current location: {current_location}
        2. Current map description: {text_map}
        3. Game progression requirements (badges, items, story events)
        
        Respond with just the location name, nothing else. Example: "Pewter City" or "Mt. Moon"."""

        # Create message for Claude
        response = self.anthropic.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": target_prompt.format(
                    current_location=self.current_location,
                    text_map=self.text_map
                )
            }]
        )

        # Extract just the location name from response
        target = " ".join([block.text for block in response.content if block.type == "text"]).strip()
        
        if target != self.target_location:
            self.target_location = target
            logger.info(f"Target Location Updated:")
            logger.info("-" * 50)
            logger.info(f"Current: {self.current_location}")
            logger.info(f"New Target: {target}")
            logger.info("-" * 50)
            
            # Get and store navigation instructions to new target
            self.navigation_instructions = self.get_navigation_instructions()
            
            # Log the full navigation plan
            logger.info("Navigation Plan Updated:")
            logger.info("=" * 50)
            logger.info(f"Current Location: {self.current_location}")
            logger.info(f"Target Location: {target}")
            logger.info(f"Navigation Instructions:\n{self.navigation_instructions}")
            logger.info("=" * 50)

    def process_tool_call(self, tool_call):
        """Process a single tool call."""
        tool_name = tool_call.name
        tool_input = tool_call.input
        
        # In quiet mode, only log at debug level
        if self.display_config['quiet_mode']:
            logger.debug(f"Processing tool call: {tool_name}")
        else:
            logger.info(f"Processing tool call: {tool_name}")

        if tool_name == "press_buttons":
            buttons = tool_input["buttons"]
            wait = tool_input.get("wait", True)
            
            # Log the button press action
            if self.display_config['quiet_mode']:
                logger.debug(f"[Buttons] Pressing: {buttons} (wait={wait})")
            else:
                logger.info(f"[Buttons] Pressing: {buttons} (wait={wait})")
            
            # Use enhanced client method to get result, state, and screenshot in one call
            response = self.client.press_buttons(
                buttons, 
                wait=wait, 
                include_state=True, 
                include_screenshot=True
            )
            
            # Extract results from response
            result = response.get('result', f"Pressed buttons: {', '.join(buttons)}")
            
            # Get game state from response or fetch it if not included
            if 'game_state' in response:
                memory_info = response['game_state'].get('game_state', '')
                
                # Update current location from game state
                old_location = self.current_location
                self.update_location_from_state(memory_info)
                
                # Log location change if it occurred
                if old_location != self.current_location:
                    logger.info("Location Changed:")
                    logger.info("-" * 50)
                    logger.info(f"From: {old_location}")
                    logger.info(f"To: {self.current_location}")
                    logger.info("-" * 50)
                
                # Update target location and get new navigation instructions if location changed
                if old_location != self.current_location:
                    self.update_target_location()
                
                if self.display_config['show_game_state']:
                    logger.info(f"[Memory State from response]")
                    logger.info(memory_info)
                else:
                    logger.debug(f"[Memory State from response]")
                    logger.debug(memory_info)
                
                collision_map = response['game_state'].get('collision_map', '')
                if collision_map and self.display_config['show_collision_map']:
                    logger.info(f"[Collision Map from response]\n{collision_map}")
                elif collision_map:
                    logger.debug(f"[Collision Map from response]\n{collision_map}")
            else:
                # Fallback to separate calls if state not included
                memory_info = self.client.get_state_from_memory()
                if self.display_config['show_game_state']:
                    logger.info(f"[Memory State after action]")
                    logger.info(memory_info)
                else:
                    logger.debug(f"[Memory State after action]")
                    logger.debug(memory_info)
                
                collision_map = self.client.get_collision_map()
                if collision_map and self.display_config['show_collision_map']:
                    logger.info(f"[Collision Map after action]\n{collision_map}")
                elif collision_map:
                    logger.debug(f"[Collision Map after action]\n{collision_map}")
            
            # Get screenshot from response or fetch it if not included
            if 'screenshot' in response:
                screenshot_b64 = response['screenshot']
            else:
                screenshot = self.client.get_screenshot()
                screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
            
            # Build response content based on display configuration
            content = [
                {"type": "text", "text": f"Pressed buttons: {', '.join(buttons)}"},
                {"type": "text", "text": "\nHere is a screenshot of the screen after your button presses:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_b64,
                    },
                }
            ]
            
            # Add game state to Claude's view if enabled
            content.append({"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"})
            
            # Return tool result as a dictionary
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": content,
            }
        elif tool_name == "navigate_to":
            row = tool_input["row"]
            col = tool_input["col"]
            
            # Log the navigation action
            if self.display_config['quiet_mode']:
                logger.debug(f"[Navigation] Navigating to: ({row}, {col})")
            else:
                logger.info(f"[Navigation] Navigating to: ({row}, {col})")
            
            # Use enhanced client method to get result, state, and screenshot in one call
            response = self.client.navigate(
                row, 
                col, 
                include_state=True, 
                include_screenshot=True
            )
            
            # Extract navigation result
            status = response.get('status', 'Unknown status')
            path = response.get('path', [])
            
            if path:
                result = f"Navigation successful: followed path with {len(path)} steps"
            else:
                result = f"Navigation failed: {status}"
            
            # Get game state from response or fetch it if not included
            if 'game_state' in response:
                memory_info = response['game_state'].get('game_state', '')
                if self.display_config['show_game_state']:
                    logger.info(f"[Memory State from response]")
                    logger.info(memory_info)
                else:
                    logger.debug(f"[Memory State from response]")
                    logger.debug(memory_info)
                
                collision_map = response['game_state'].get('collision_map', '')
                if collision_map and self.display_config['show_collision_map']:
                    logger.info(f"[Collision Map from response]\n{collision_map}")
                elif collision_map:
                    logger.debug(f"[Collision Map from response]\n{collision_map}")
            else:
                # Fallback to separate calls if state not included
                memory_info = self.client.get_state_from_memory()
                if self.display_config['show_game_state']:
                    logger.info(f"[Memory State after action]")
                    logger.info(memory_info)
                else:
                    logger.debug(f"[Memory State after action]")
                    logger.debug(memory_info)
                
                collision_map = self.client.get_collision_map()
                if collision_map and self.display_config['show_collision_map']:
                    logger.info(f"[Collision Map after action]\n{collision_map}")
                elif collision_map:
                    logger.debug(f"[Collision Map after action]\n{collision_map}")
            
            # Get screenshot from response or fetch it if not included
            if 'screenshot' in response:
                screenshot_b64 = response['screenshot']
            else:
                screenshot = self.client.get_screenshot()
                screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
            
            # Build response content based on display configuration
            content = [
                {"type": "text", "text": f"Navigation result: {result}"},
                {"type": "text", "text": "\nHere is a screenshot of the screen after navigation:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_b64,
                    },
                }
            ]
            
            # Add game state to Claude's view if enabled
            content.append({"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"})
            
            # Return tool result as a dictionary
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": content,
            }
        else:
            logger.error(f"Unknown tool called: {tool_name}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}
                ],
            }

    def run(self, num_steps=1):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """
        if self.display_config['quiet_mode']:
            logger.debug(f"Starting agent loop for {num_steps} steps")
        else:
            logger.info(f"Starting agent loop for {num_steps} steps")

        # Log initial text map state
        if self.text_map:
            logger.info("Current Text Map:")
            logger.info("-" * 50)
            logger.info(self.text_map)
            logger.info("-" * 50)

        steps_completed = 0
        while self.running and steps_completed < num_steps:
            try:
                messages = copy.deepcopy(self.message_history)

                if len(messages) >= 3:
                    if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
                    
                    if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                        messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}

                # Create system prompt with current text map, location info, and navigation instructions
                system_prompt = self.SYSTEM_PROMPT.format(
                    text_map=self.text_map,
                    current_location=self.current_location,
                    target_location=self.target_location,
                    navigation_instructions=self.navigation_instructions
                )
                

                # Get model response with updated system prompt
                response = self.anthropic.messages.create(
                    model=MODEL_NAME,
                    max_tokens=MAX_TOKENS,
                    system=system_prompt,
                    messages=messages,
                    tools=self.AVAILABLE_TOOLS,
                    temperature=TEMPERATURE,
                )

                # Log token usage
                if self.display_config['quiet_mode']:
                    logger.debug(f"Response usage: {response.usage}")
                else:
                    logger.info(f"Response usage: {response.usage}")

                # Extract tool calls
                tool_calls = [
                    block for block in response.content if block.type == "tool_use"
                ]

                # Display the model's reasoning
                for block in response.content:
                    if block.type == "text":
                        # Claude's thoughts should always be visible, even in quiet mode
                        logger.info(f"[Claude] {block.text}")
                    elif block.type == "tool_use":
                        # Tool calls should be visible at info level by default
                        if self.display_config['quiet_mode']:
                            logger.info(f"[Claude Action] Using tool: {block.name} with input: {block.input}")
                        else:
                            logger.info(f"[Tool Use] {block.name} with input: {block.input}")

                # Process tool calls
                if tool_calls:
                    # Add assistant message to history
                    assistant_content = []
                    for block in response.content:
                        if block.type == "text":
                            assistant_content.append({"type": "text", "text": block.text})
                        elif block.type == "tool_use":
                            assistant_content.append({"type": "tool_use", **dict(block)})
                    
                    self.message_history.append(
                        {"role": "assistant", "content": assistant_content}
                    )
                    
                    # Process tool calls and create tool results
                    tool_results = []
                    for tool_call in tool_calls:
                        tool_result = self.process_tool_call(tool_call)
                        tool_results.append(tool_result)
                    
                    # Add tool results to message history
                    self.message_history.append(
                        {"role": "user", "content": tool_results}
                    )

                    # Update the text map after processing tool calls
                    self.update_text_map()

                    # Check if we need to summarize the history
                    if len(self.message_history) >= self.max_history:
                        self.summarize_history()

                steps_completed += 1
                if self.display_config['quiet_mode']:
                    logger.debug(f"Completed step {steps_completed}/{num_steps}")
                else:
                    logger.info(f"Completed step {steps_completed}/{num_steps}")

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                raise e

        if not self.running:
            self.client.stop()

        return steps_completed
    
    def update_text_map(self):
        """Update the text map with the new information."""
        # Get current screenshot
        screenshot = self.client.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
        
        # SLAM map processing disabled for now
        # try:
        #     # Convert PIL image to OpenCV format
        #     frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        #     
        #     # Update SLAM map with new frame
        #     self.slam_map.add_frame(frame)
        #     
        #     # Get current map state and save it
        #     current_map = self.slam_map.get_map()
        #     cv2.imwrite("internal_map.png", current_map)
        #     
        # except Exception as e:
        #     logger.error(f"Error processing screenshot for SLAM: {e}")
        
        # Format the observation prompt with the current text map
        formatted_prompt = self.OBSERVATION_PROMPT.format(self=self)
        
        # Create message with both text and image
        updation_message = self.anthropic.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": formatted_prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64
                        }
                    }
                ]
            }]
        )
        
        # Update the text map with Claude's response
        response_text = " ".join([block.text for block in updation_message.content if block.type == "text"])
        self.text_map = response_text

        # Log the updated text map - always show at info level as it's important for tracking
        logger.info("Updated Text Map:")
        logger.info("-" * 50)
        logger.info(self.text_map)
        logger.info("-" * 50)

    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""
        if self.display_config['quiet_mode']:
            logger.debug(f"[Agent] Generating conversation summary...")
        else:
            logger.info(f"[Agent] Generating conversation summary...")
        
        # Get a new screenshot for the summary
        screenshot = self.client.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
        
        # Create messages for the summarization request - pass the entire conversation history
        messages = copy.deepcopy(self.message_history) 

        if len(messages) >= 3:
            if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            
            if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        messages += [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.SUMMARY_PROMPT,
                    }
                ],
            }
        ]
        
        # Get summary from Claude
        response = self.anthropic.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=self.SYSTEM_PROMPT,
            messages=messages,
            temperature=TEMPERATURE
        )
        
        # Extract the summary text
        summary_text = " ".join([block.text for block in response.content if block.type == "text"])
        
        # Log the summary - use info level even in quiet mode as it's important
        logger.info(f"[Claude Summary] Game Progress Summary:")
        logger.info(f"{summary_text}")
        
        # Replace message history with just the summary
        self.message_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}"
                    },
                    {
                        "type": "text",
                        "text": "\n\nCurrent game screenshot for reference:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "You were just asked to summarize your playthrough so far, which is the summary you see above. You may now continue playing by selecting your next action."
                    },
                ]
            }
        ]
        
        if self.display_config['quiet_mode']:
            logger.debug(f"[Agent] Message history condensed into summary.")
        else:
            logger.info(f"[Agent] Message history condensed into summary.")
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        self.client.stop()


class SLAMMap:
    def __init__(self, canvas_size=(5000, 5000), initial_offset=(2500, 2500)):
        """
        Initializes a large blank canvas and a global coordinate system.
        Instead of a full transformation matrix, we keep track of a global (x,y)
        position indicating where the top-left corner of the incoming screenshot
        should be placed.
        """
        self.canvas_size = canvas_size
        self.canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        # Global position (top-left coordinate) where the next screenshot will be pasted
        self.global_position = np.array(initial_offset, dtype=np.float64)
        self.last_frame = None

    def add_frame(self, frame):
        """
        Instead of warping the image, this method computes a translation offset
        relative to the previous frame and pastes the new screenshot at the updated
        global coordinate.
        """
        if self.last_frame is None:
            # For the first frame, paste it at the initial global position
            self._paste_on_canvas(frame, self.global_position)
            self.last_frame = frame
            logger.debug("Initialized map with first frame at position", self.global_position)
            return

        translation = compute_translation(self.last_frame, frame)
        if translation is None:
            logger.warning("Could not compute translation for the new frame. Skipping frame.")
            return

        dx, dy = translation
        # Update the global position by adding the computed translation
        self.global_position += np.array([dx, dy])
        self._paste_on_canvas(frame, self.global_position)
        self.last_frame = frame
        logger.debug("Frame added at position", self.global_position)

    def _paste_on_canvas(self, frame, top_left):
        """
        Pastes the given frame onto the canvas at the position specified by top_left.
        Handles boundary conditions if the frame extends beyond the canvas edges.
        """
        x, y = int(top_left[0]), int(top_left[1])
        h, w = frame.shape[:2]
        canvas_h, canvas_w = self.canvas.shape[:2]
        
        # Determine the region on the canvas where the image will be placed
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, canvas_w)
        y2 = min(y + h, canvas_h)
        
        # Determine the corresponding region in the frame
        fx1 = 0 if x >= 0 else -x
        fy1 = 0 if y >= 0 else -y
        fx2 = fx1 + (x2 - x1)
        fy2 = fy1 + (y2 - y1)
        
        self.canvas[y1:y2, x1:x2] = frame[fy1:fy2, fx1:fx2]

    def get_map(self):
        """Returns the current internal map (the stitched canvas)."""
        return self.canvas


def compute_translation(img1, img2):
    """
    Compute the translation vector (dx, dy) between img1 and img2.
    This function uses ORB features and RANSAC (via a homography) but then
    extracts only the translation components (assuming minimal rotation/scale).
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector and compute keypoints/descriptors
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return None

    # Match descriptors using Brute-Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 4:
        return None

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute homography using RANSAC
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        return None
    
    # Extract translation components
    tx = H[0, 2]
    ty = H[1, 2]
    return tx, ty


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run a Pokemon Game Server Agent')
    parser.add_argument('--snapshot-id', type=str, required=True, help='Morph snapshot ID to run')
    parser.add_argument('--api-key', type=str, help='Morph API key (defaults to MORPH_API_KEY env var)')
    parser.add_argument('--steps', type=int, default=10, help='Number of steps to run (default: 10)')
    parser.add_argument('--max-history', type=int, default=30, help='Maximum history size before summarizing (default: 30)')
    
    # Add verbosity and display options
    parser.add_argument('--verbose', '-v', action='count', default=0, 
                        help='Increase output verbosity (can be used multiple times, e.g. -vv)')
    parser.add_argument('--show-game-state', action='store_true', 
                        help='Show full game state information in the logs')
    parser.add_argument('--show-collision-map', action='store_true', 
                        help='Show collision map in the logs')
    parser.add_argument('--log-file', type=str, 
                        help='Path to log file. If not provided, logs will only go to stderr')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Only show Claude\'s thoughts and actions, minimal logging')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Configure logging based on command line arguments
    log_handlers = []
    
    # Set up console handler with formatting
    console_handler = logging.StreamHandler()
    if args.quiet:
        console_format = "%(message)s"  # Minimal format for quiet mode
    else:
        console_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    console_handler.setFormatter(logging.Formatter(console_format))
    log_handlers.append(console_handler)
    
    # Add file handler if log file specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        # Full detailed format for log files
        file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file_handler.setFormatter(logging.Formatter(file_format))
        log_handlers.append(file_handler)
    
    # Set log level based on verbosity
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose == 0:
        log_level = logging.INFO
    elif args.verbose == 1:
        log_level = logging.DEBUG
    else:  # args.verbose >= 2
        log_level = logging.DEBUG  # Maximum verbosity
    
    # Configure the root logger
    logging.basicConfig(level=log_level, handlers=log_handlers, force=True)
    
    # Create a rich console for nice output
    from rich.console import Console
    console = Console()
    
    console.print(f"Starting Pokemon Game Server Agent from snapshot {args.snapshot_id}")
    console.print(f"Will run for {args.steps} steps with max history of {args.max_history}")
    if not args.quiet:
        console.print(f"Log level: {'QUIET' if args.quiet else logging.getLevelName(log_level)}")
        if args.show_game_state:
            console.print("Game state display: Enabled")
        if args.show_collision_map:
            console.print("Collision map display: Enabled")
        if args.log_file:
            console.print(f"Logging to file: {args.log_file}")
    console.print("=" * 50)
    
    # Initialize Morph client and start instance
    from morphcloud.api import MorphCloudClient
    
    morph_client = MorphCloudClient(api_key=args.api_key)
    
    # Start instance from snapshot
    console.print("Starting instance from snapshot...")
    instance = morph_client.instances.start(args.snapshot_id)
    
    # Wait for instance to be ready
    console.print("Waiting for instance to be ready...")
    instance.wait_until_ready()
    
    # Get the instance URL
    instance_url = next(service.url for service in instance.networking.http_services if service.name == "web")
    
    remote_desktop_url = next(service.url for service in instance.networking.http_services if service.name == "novnc")
    
    novnc_url = f"{remote_desktop_url}/vnc_lite.html"
    console.print(f"Pokemon remote desktop available at: {novnc_url}")

    # Open the NoVNC URL automatically in the default browser
    import webbrowser
    webbrowser.open(novnc_url)
    
    # Create a "game display" configuration object to pass to the agent
    display_config = {
        'show_game_state': args.show_game_state or args.verbose > 0,
        'show_collision_map': args.show_collision_map or args.verbose > 1,
        'quiet_mode': args.quiet
    }
    
    # Run agent with the instance URL
    console.print("Initializing agent...")
    try:
        agent = ServerAgent(
            server_host=instance_url,
            server_port=None,  # Not needed since URL already includes the port
            max_history=args.max_history,
            display_config=display_config
        )
        
        console.print("✅ Agent initialized successfully!")
        console.print("=" * 50)
        
        # Run the agent
        console.print(f"Starting agent loop for {args.steps} steps...")
        steps_completed = agent.run(num_steps=args.steps)
        
        console.print("=" * 50)
        console.print(f"✅ Agent completed {steps_completed} steps")
        
    except ConnectionError as e:
        console.print(f"❌ Connection error: {e}")
        console.print(f"Make sure the server is running on the instance")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("Received keyboard interrupt, stopping agent")
    except Exception as e:
        console.print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        if 'agent' in locals():
            agent.stop()
        
        # Stop the Morph instance
        console.print("Stopping Morph instance...")
        instance.stop()

if __name__ == "__main__":
    main()
