#!/usr/bin/env python
"""
Content Writing Crew - Main Entry Point

This script provides a command-line interface to run the content writing crew
with custom parameters.

Usage:
    python main.py --topic "Your Topic" --goal "Your Content Goal" --words 2000

Example:
    python main.py --topic "Machine Learning in Healthcare" --goal "Educate healthcare professionals" --words 1500
"""

import sys
import argparse
from pathlib import Path
from crew import ContentWritingCrew


def setup_output_directory():
    """Ensure the output directory exists"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Content Writing Crew - AI-powered content creation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --topic "AI in Education"
  python main.py --topic "Sustainable Energy" --goal "Raise awareness" --words 2500
  python main.py --interactive
        """
    )

    parser.add_argument(
        '--topic',
        type=str,
        help='The topic to write about'
    )

    parser.add_argument(
        '--goal',
        type=str,
        default='Provide comprehensive, actionable information to readers',
        help='The content goal/objective (default: provide actionable information)'
    )

    parser.add_argument(
        '--words',
        type=int,
        default=2000,
        help='Target word count for the article (default: 2000)'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode to input parameters'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def get_interactive_inputs():
    """Get inputs from user interactively"""
    print("=" * 70)
    print("  Content Writing Crew - Interactive Mode")
    print("=" * 70)
    print()

    topic = input("📝 Enter the topic for your article: ").strip()
    if not topic:
        print("❌ Topic cannot be empty!")
        sys.exit(1)

    goal = input("\n🎯 Enter the content goal (or press Enter for default): ").strip()
    if not goal:
        goal = "Provide comprehensive, actionable information to readers"

    word_count_input = input("\n📏 Enter target word count (or press Enter for 2000): ").strip()
    try:
        word_count = int(word_count_input) if word_count_input else 2000
    except ValueError:
        print("⚠️  Invalid word count, using default: 2000")
        word_count = 2000

    return {
        'topic': topic,
        'content_goal': goal,
        'word_count': word_count
    }


def display_config(inputs):
    """Display the current configuration"""
    print("\n" + "=" * 70)
    print("  Configuration")
    print("=" * 70)
    print(f"📌 Topic:       {inputs['topic']}")
    print(f"🎯 Goal:        {inputs['content_goal']}")
    print(f"📏 Word Count:  {inputs['word_count']}")
    print("=" * 70)
    print()


def main():
    """Main execution function"""
    args = parse_arguments()

    # Setup
    setup_output_directory()

    # Get inputs
    if args.interactive:
        inputs = get_interactive_inputs()
    else:
        if not args.topic:
            print("❌ Error: --topic is required (or use --interactive mode)")
            print("Run with --help for usage information")
            sys.exit(1)

        inputs = {
            'topic': args.topic,
            'content_goal': args.goal,
            'word_count': args.words
        }

    # Display configuration
    display_config(inputs)

    # Run the crew
    print("🚀 Starting Content Writing Crew...")
    print("⏳ This may take several minutes depending on the complexity...\n")

    try:
        crew = ContentWritingCrew().crew()
        result = crew.kickoff(inputs=inputs)

        print("\n" + "=" * 70)
        print("  ✅ Content Writing Crew Completed Successfully!")
        print("=" * 70)
        print(f"\n📁 Output files saved to: ./output/")
        print(f"   - Draft article: output/draft_article.md")
        print(f"   - Final article: output/final_article.md")
        print("\n" + "=" * 70)

        if args.verbose:
            print("\n📄 Final Result:")
            print(result)

        return result

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
