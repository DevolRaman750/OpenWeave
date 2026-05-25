#!/usr/bin/env python3
"""
RepoMaster Multi-Agent System Launcher

This file is the main startup entry point for RepoMaster's Multi-Agent Intelligence System, 
supporting multiple access interfaces:
1. frontend: Interactive Multi-Agent Dashboard
2. backend: Multi-Agent Service Interface
   - unified: Unified Multi-Agent Interface (Recommended - automatic agent orchestration)
   - deepsearch: Direct Deep Search Agent access
   - general_assistant: Direct Programming Assistant Agent access  
   - repository_agent: Direct Repository Exploration Agent access

Usage:
    python launcher.py --mode frontend                              # Multi-Agent Dashboard
    python launcher.py --mode backend --backend-mode unified        # Unified Multi-Agent Interface
    python launcher.py --mode backend --backend-mode deepsearch     # Deep Search Agent
    python launcher.py --mode backend --backend-mode general_assistant  # Programming Assistant Agent
    python launcher.py --mode backend --backend-mode repository_agent   # Repository Exploration Agent
    python launcher.py --help  # View all options
"""

import os
import sys
import logging
import asyncio
import subprocess
from pathlib import Path


def configure_stdio_encoding():
    """Keep launcher status output from crashing on Windows cp1252 consoles."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


configure_stdio_encoding()


from configs.mode_config import ModeConfigManager, create_argument_parser, print_config_info
try:
    from src.frontend.terminal_show import (
        print_repomaster_cli, print_startup_banner, print_environment_status,
        print_api_config_status, print_launch_config, print_service_starting,
        print_unified_mode_welcome, print_mode_welcome, print_progressive_startup_panel, print_repomaster_title
    )
except Exception:
    def print_repomaster_cli(*args, **kwargs):
        return None

    def print_startup_banner(*args, **kwargs):
        return None

    def print_environment_status(*args, **kwargs):
        return None

    def print_api_config_status(*args, **kwargs):
        return None

    def print_launch_config(*args, **kwargs):
        return None

    def print_service_starting(*args, **kwargs):
        return None

    def print_unified_mode_welcome(work_dir, *args, **kwargs):
        print(f"\nRepoMaster Unified Mode (ACM auto)\nWorking directory: {work_dir}")

    def print_mode_welcome(title, work_dir, features=None, instructions=None, *args, **kwargs):
        print(f"\n{title}\nWorking directory: {work_dir}")
        if features:
            for feature in features:
                print(feature)
        if instructions:
            for instruction in instructions:
                print(instruction)

    def print_progressive_startup_panel(*args, **kwargs):
        return None

    def print_repomaster_title(*args, **kwargs):
        print("\n=== RepoMaster ===")

def setup_logging(log_level: str):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Reduce warning messages from third-party libraries
    if log_level.upper() != 'DEBUG':
        logging.getLogger('autogen.oai.client').setLevel(logging.ERROR)
        logging.getLogger('langchain_community.utils.user_agent').setLevel(logging.ERROR)

def setup_environment():
    """Setup environment variables"""
    # Setup PYTHONPATH
    current_dir = Path(__file__).parent.absolute()
    python_path = os.environ.get('PYTHONPATH', '')
    if str(current_dir) not in python_path:
        os.environ['PYTHONPATH'] = f"{current_dir}:{python_path}" if python_path else str(current_dir)
    
    # Load environment variables
    from dotenv import load_dotenv
    
    # Check for configuration files
    env_file = current_dir / "configs" / ".env"
    env_example_file = current_dir / "configs" / "env.example"
    
    # If .env doesn't exist but env.example does, provide helpful guidance
    if not env_file.exists() and env_example_file.exists():
        print("⚠️  Configuration file not found!")
        print(f"📝 Please copy the example configuration file:")
        print(f"   cp {env_example_file} {env_file}")
        print(f"   Then edit {env_file} with your API keys")
        print("💡 See README.md or USAGE.md for detailed setup instructions")
        return False
    
    # Load environment variables if .env exists
    if env_file.exists():
        load_dotenv(env_file, override=True)
        
        # Check for required API keys
        missing_keys = []
        required_keys = ['SERPER_API_KEY', 'JINA_API_KEY']
        
        for key in required_keys:
            if not os.environ.get(key):
                missing_keys.append(key)
        
        if missing_keys:
            print(f"⚠️  Missing required API keys in .env file: {', '.join(missing_keys)}")
            print(f"📝 Please edit {env_file} and add the missing keys")
            print("💡 See README.md or USAGE.md for API key setup instructions")
            return False
            
        return True
    
    # Fallback to system environment variables
    print("⚠️  .env file not found, checking system environment variables...")
    missing_keys = []
    required_keys = ['SERPER_API_KEY', 'JINA_API_KEY']
    
    for key in required_keys:
        if not os.environ.get(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing required environment variables: {', '.join(missing_keys)}")
        if env_example_file.exists():
            print(f"📝 Please create configuration file:")
            print(f"   cp {env_example_file} {env_file}")
            print(f"   Then edit {env_file} with your API keys")
        print("💡 See README.md or USAGE.md for setup instructions")
        return False
        
    print("✅ Using system environment variables")
    return True

def run_frontend_mode(config_manager: ModeConfigManager):
    """Run frontend mode"""
    config = config_manager.config

    frontend_entry = Path("src/frontend/app_autogen_enhanced.py")
    if not frontend_entry.exists():
        raise RuntimeError(
            "Frontend directory is not available in this RepoMaster build. "
            "Use backend unified mode with ACM input instead."
        )
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/frontend/app_autogen_enhanced.py",
        "--server.port", str(config.streamlit_port),
        "--server.address", config.streamlit_host,
        "--server.fileWatcherType", config.file_watcher_type,
        "--server.maxUploadSize", str(config.max_upload_size)
    ]
    
    print(f"\n🌐 Access URL: http://{config.streamlit_host}:{config.streamlit_port}")
    print(f"⚡ Execute command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Frontend service stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend startup failed: {e}")
        sys.exit(1)

def run_backend_mode(config_manager: ModeConfigManager):
    """Run backend mode"""
    config = config_manager.config
    
    if config.backend_mode == "deepsearch":
        run_deepsearch_mode(config_manager)
    elif config.backend_mode == "general_assistant":
        run_general_assistant_mode(config_manager)
    elif config.backend_mode == "repository_agent":
        run_repository_agent_mode(config_manager)
    elif config.backend_mode == "unified":
        run_unified_mode(config_manager)
    else:
        raise ValueError(f"Unsupported backend mode: {config.backend_mode}")

def run_deepsearch_mode(config_manager: ModeConfigManager):
    """Run Deep Search Agent (direct access to deep search capabilities)"""
    
    # Import deep search agent and conversation manager
    from src.services.agents.deep_search_agent import AutogenDeepSearchAgent
    from src.core.conversation_manager import ConversationManager, get_user_id_for_cli
    
    # Get configuration
    llm_config = config_manager.get_llm_config(config_manager.config.api_type)
    execution_config = config_manager.get_execution_config()
    
    # Create deep search agent
    agent = AutogenDeepSearchAgent(
        llm_config=llm_config,
        code_execution_config=execution_config
    )
    
    # Create conversation manager
    user_id = get_user_id_for_cli()
    conversation = ConversationManager(user_id, "deepsearch")
    
    # Display beautiful welcome message
    features = [
        "🔍 Advanced search & query optimization",
        "🌐 Real-time web information retrieval"
    ]
    
    instructions = [
        "• Enter search question or research topic",
        "• Enter 'quit' to exit, 'history'/'clear' to view/clear chat history"
    ]
    
    print_mode_welcome("🔍 Deep Search Agent Ready!", execution_config['work_dir'], features, instructions)
    
    try:
        while True:
            query = input("\n🤔 Please enter search question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query.lower() in ['history', 'h']:
                conversation.show_history()
                continue
                
            if query.lower() in ['clear', 'c']:
                conversation.clear_conversation()
                continue
            
            if not query:
                continue
            
            # Get optimized prompt with conversation context
            optimized_query = conversation.get_optimized_prompt(query)
            conversation.add_message("user", query)
            
            print("🔍 Searching...")
            result = asyncio.run(agent.deep_search(optimized_query))
            conversation.add_message("assistant", result)
            print(f"\n📋 Search results:\n{result}\n")
            
    except KeyboardInterrupt:
        print("\n👋 Deep Search Agent service stopped")

def run_general_assistant_mode(config_manager: ModeConfigManager):
    """Run Programming Assistant Agent (direct access to programming assistance capabilities)"""
    
    # Import RepoMaster agent and conversation manager
    from src.core.agent_scheduler import RepoMasterAgent
    from src.core.conversation_manager import ConversationManager, get_user_id_for_cli
    
    # Get configuration
    llm_config = config_manager.get_llm_config(config_manager.config.api_type)
    execution_config = config_manager.get_execution_config()
    
    # Create RepoMaster agent
    agent = RepoMasterAgent(
        llm_config=llm_config,
        code_execution_config=execution_config
    )
    
    # Create conversation manager
    user_id = get_user_id_for_cli()
    conversation = ConversationManager(user_id, "general_assistant")
    
    # Display beautiful welcome message
    features = [
        "💻 General purpose programming assistance",
        "🔧 Code writing, debugging and optimization",
        "📚 Algorithm implementation & debugging help"
    ]
    
    instructions = [
        "• Describe programming task or ask questions",
        "• Enter 'quit' to exit, 'history'/'clear' to view/clear chat history"
    ]
    
    print_mode_welcome("Programming Assistant Ready!", execution_config['work_dir'], features, instructions)
    
    try:
        while True:
            task = input("\n💻 Please describe your programming task: ").strip()
            if task.lower() in ['quit', 'exit', 'q']:
                break
            
            if task.lower() in ['history', 'h']:
                conversation.show_history()
                continue
                
            if task.lower() in ['clear', 'c']:
                conversation.clear_conversation()
                continue
            
            if not task:
                continue
            
            # Get optimized prompt with conversation context
            optimized_task = conversation.get_optimized_prompt(task)
            conversation.add_message("user", task)
            
            print("🔧 Processing...")
            # Call run_general_code_assistant
            result = agent.run_general_code_assistant(
                task_description=optimized_task,
                work_directory=execution_config.get("work_dir")
            )
            conversation.add_message("assistant", result)
            print_repomaster_title()
            print(f"\n📋 Task result:\n{result}\n")
            
    except KeyboardInterrupt:
        print("\n👋 Programming Assistant Agent service stopped")

def run_repository_agent_mode(config_manager: ModeConfigManager):
    """Run Repository Exploration Agent (direct access to repository exploration and task execution)"""
    
    # Import RepoMaster agent and conversation manager
    from src.core.agent_scheduler import RepoMasterAgent
    from src.core.conversation_manager import ConversationManager, get_user_id_for_cli
    
    # Get configuration
    llm_config = config_manager.get_llm_config(config_manager.config.api_type)
    execution_config = config_manager.get_execution_config()
    
    # Create RepoMaster agent
    agent = RepoMasterAgent(
        llm_config=llm_config,
        code_execution_config=execution_config
    )
    
    # Create conversation manager
    user_id = get_user_id_for_cli()
    conversation = ConversationManager(user_id, "repository_agent")
    
    # Display beautiful welcome message
    features = [
        "📁 Repository analysis & structure modeling",
        "🔧 Autonomous code exploration and execution"
    ]
    
    instructions = [
        "• Provide task description and repository (GitHub URL or local path)",
        "• Optional: add input data files | Enter 'quit' to exit, 'history' to view chat history, 'clear' to clear history"
    ]
    
    print_mode_welcome("Repository Agent Ready!", execution_config['work_dir'], features, instructions)
    
    try:
        while True:
            task_description = input("\n📝 Please describe your task: ").strip()
            if task_description.lower() in ['quit', 'exit', 'q']:
                break
            
            if task_description.lower() in ['history', 'h']:
                conversation.show_history()
                continue
                
            if task_description.lower() in ['clear', 'c']:
                conversation.clear_conversation()
                continue
            
            if not task_description:
                continue
            
            repository = input("📁 Please enter repository path or URL: ").strip()
            if not repository:
                print("❌ Repository path cannot be empty")
                continue
            
            # Optional: input data
            use_input_data = input("🗂️  Do you need to provide input data files? (y/N): ").strip().lower()
            input_data = None
            
            if use_input_data in ['y', 'yes']:
                input_path = input("📂 Please enter data file path: ").strip()
                if input_path and os.path.exists(input_path):
                    input_data = f'[{{"path": "{input_path}", "description": "User provided input data"}}]'
                else:
                    print("⚠️  Input path invalid, will ignore input data")
            
            # Get optimized prompt with conversation context
            optimized_task = conversation.get_optimized_prompt(task_description)
            conversation.add_message("user", f"Task: {task_description}\nRepository: {repository}")
            
            print("🔧 Processing repository task...")
            
            # Call run_repository_agent
            result = agent.run_repository_agent(
                task_description=optimized_task,
                repository=repository,
                input_data=input_data
            )
            conversation.add_message("assistant", result)
            print_repomaster_title()
            print(f"\n📋 Task result:\n{result}\n")
            
    except KeyboardInterrupt:
        print("\n👋 Repository Exploration Agent service stopped")

def run_unified_mode(config_manager: ModeConfigManager):
    """Run Unified mode in ACM-only automation (no external task input)."""

    # Import RepoMaster agent
    from src.core.agent_scheduler import RepoMasterAgent

    # Get configuration
    llm_config = config_manager.get_llm_config(config_manager.config.api_type)
    execution_config = config_manager.get_execution_config()

    # Create RepoMaster agent
    agent = RepoMasterAgent(
        llm_config=llm_config,
        code_execution_config=execution_config
    )

    # Display beautiful welcome message (unified mode specific)
    print_unified_mode_welcome(execution_config['work_dir'])

    print("\n🔧 ACM auto-run started...")
    print("   📌 Query source: input_acm.json -> search_intent.repomaster_query")
    print("   📌 Context source: compacted full input_acm.json")

    try:
        # External task text is intentionally ignored in ACM mode.
        result = agent.solve_task_with_repo("ACM_AUTO_QUERY")
        print_repomaster_title()
        print("\n📋 ACM repository search result:")
        print(result)
    except Exception as e:
        from src.utils.error_artifacts import write_error_artifact
        import traceback
        artifact_path = getattr(e, "_repomaster_artifact_path", "")
        if not artifact_path:
            artifact_path = write_error_artifact(
                component="scheduler",
                operation="run_unified_mode",
                error=e,
                context={
                    "backend_mode": "unified",
                    "work_dir": execution_config.get("work_dir"),
                },
            )
            setattr(e, "_repomaster_artifact_path", artifact_path)

        print(traceback.format_exc())
        if artifact_path:
            print(f"🧾 Error artifact: {artifact_path}")
        print(f"\n❌ ACM auto-run error: {str(e)}")
        print("   💡 Ensure input_acm.json has search_intent.repomaster_query")
        raise

def check_api_configuration() -> bool:
    """Check API configuration status"""
    try:
        from configs.oai_config import validate_and_get_fallback_config
        config_name, api_config = validate_and_get_fallback_config()
        return True
            
    except ImportError:
        print("⚠️  oai_config not found, skip configuration check")
        return False
    except Exception as e:
        print(f"⚠️  Configuration check error: {e}")
        return False

def show_available_modes():
    """Display available Multi-Agent system interfaces"""
    print("""
🤖 RepoMaster Multi-Agent System Access Interfaces:

1. Multi-Agent Dashboard (Visual Interface)
   - Interactive web interface with agent collaboration visualization
   - Real-time multi-agent coordination display
    - Command: python launcher.py --mode frontend (only if frontend files exist)

2. Multi-Agent Service Interface (Backend)
   - unified: Unified Multi-Agent Interface ⭐ Recommended
     python launcher.py --mode backend --backend-mode unified
    🧠 ACM auto mode: uses input_acm.json repomaster_query + compacted full ACM context (no user query prompt)
     
   - deepsearch: Deep Search Agent (Direct Access)
     python launcher.py --mode backend --backend-mode deepsearch
     🔍 Advanced web search, data analysis, and information synthesis
     
   - general_assistant: Programming Assistant Agent (Direct Access)
     python launcher.py --mode backend --backend-mode general_assistant
     💻 Code generation, algorithm implementation, and debugging assistance
     
   - repository_agent: Repository Exploration Agent (Direct Access)
     python launcher.py --mode backend --backend-mode repository_agent
     🏗️ Repository exploration, task execution, and system orchestration

🔧 Advanced Options:
    --api-type: Specify API type (nvidia, basic, openai, claude, etc.)
   --temperature: Set model temperature (0.0-2.0)
   --work-dir: Specify working directory
   --log-level: Set log level (DEBUG, INFO, WARNING, ERROR)
   --skip-config-check: Skip API configuration check

📖 Get complete help: python launcher.py --help

💡 First time use? Reference: USAGE.md
""")

def main():
    """Main function"""
    # Print RepoMaster CLI logo first
    # print_repomaster_cli()
    
    # Setup environment
    env_loaded = setup_environment()
    
    # Prepare environment status
    env_status = {
        'success': env_loaded,
        'file': str(Path(__file__).parent / "configs" / ".env") if env_loaded else None
    }
    
    if not env_loaded:
        # Show error and exit
        dummy_config = type('Config', (), {
            'mode': 'error',
            'work_dir': Path(__file__).parent,
            'log_level': 'INFO'
        })()
        api_status = {'success': False}
        print_progressive_startup_panel(env_status, api_status, dummy_config)
        sys.exit(1)
    
    # Check if help or mode information is requested
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['--modes', '--list-modes']):
        show_available_modes()
        return
    
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        setup_logging(args.log_level)
        
        # Configuration check (unless user explicitly skips)
        api_status = {'success': False}
        if not getattr(args, 'skip_config_check', False):
            api_config_success = check_api_configuration()
            if api_config_success:
                # Get config info for display
                try:
                    from configs.oai_config import validate_and_get_fallback_config
                    config_info = validate_and_get_fallback_config()
                    if config_info:
                        config_name, config_details = config_info
                        model = config_details.get('config_list', [{}])[0].get('model', 'N/A')
                        api_status = {
                            'success': True,
                            'provider': args.api_type.title(),
                            'model': model
                        }
                except Exception:
                    api_status = {'success': True, 'provider': args.api_type.title()}
            else:
                api_status = {'success': False}
                # Show error panel and exit
                dummy_config = type('Config', (), {
                    'mode': 'error',
                    'work_dir': Path(__file__).parent,
                    'log_level': args.log_level
                })()
                print_progressive_startup_panel(env_status, api_status, dummy_config)
                sys.exit(1)
        else:
            api_status = {'success': True, 'provider': 'Skipped (user choice)'}
        
        # Create configuration manager
        config_manager = ModeConfigManager.from_args(args)
        
        # Print optimized startup sequence  
        print_progressive_startup_panel(env_status, api_status, config_manager.config)
        
        # Start corresponding service based on mode
        if args.mode == 'frontend':
            run_frontend_mode(config_manager)
        elif args.mode == 'backend':
            run_backend_mode(config_manager)
        else:
            raise ValueError(f"Unsupported running mode: {args.mode}")
            
    except KeyboardInterrupt:
        print("\n👋 Program interrupted by user")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"Startup failed: {e}")
        print(f"❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
