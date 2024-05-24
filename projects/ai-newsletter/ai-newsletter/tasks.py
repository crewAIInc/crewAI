from datetime import datetime
from crewai import Task

class AINewsLetterTasks():
    def fetch_news_task(self, agent):
        return Task(
            description=f'Fetch the most relevant top AI news stories from the past 24 hours. The current time is {datetime.now()}.',
            agent=agent,
            async_execution=True,
            expected_output="""A list of top AI news story titles, URLs, and a brief summary for each story from the past 24 hours. 
                Example Output: 
                [
                    {  'title': 'AI takes spotlight in Super Bowl commercials', 
                    'url': 'https://example.com/story1', 
                    'summary': 'AI made a splash in this year\'s Super Bowl commercials...'
                    }, 
                    {{...}}
                ]
            """
        )

    def analyze_news_task(self, agent, context):
        return Task(
            description='Analyze each news story and ensure there are at least 5 well-formatted articles',
            agent=agent,
            async_execution=True,
            context=context,
            expected_output="""A markdown-formatted analysis for each news story, including a rundown, detailed bullet points, 
                and a "Why it matters" section. There should be at least 5 articles, each following the proper format.
                Example Output: 
                '## AI takes spotlight in Super Bowl commercials\\n\\n

                **The Rundown:**
                AI made a splash in this year\'s Super Bowl commercials...\\n\\n

                **The details:**\\n\\n

                - Microsoft's Copilot spot showcased its AI assistant...\\n\\n

                **Why it matters:** While AI-related ads have been rampant over the last year, its Super Bowl presence is a big mainstream moment.\\n\\n'
            """
        )

    def compile_newsletter_task(self, agent, context, callback_function):
        return Task(
            description='Compile the newsletter',
            agent=agent,
            context=context,
            expected_output="""A complete newsletter in markdown format, with a consistent style and layout.
                Example Output: 
                '# 🌞 Good Morning Subscribers!\\n\\n
                Welcome to today\'s edition of the AI Newsletter, where we bring you the latest and greatest in AI, served with a touch of wit and wisdom. Let\'s dive into the exciting world of AI!\\n\\n

                # 📰 AI INSIGHTS - Your Daily AI News Update:\\n\\n
                1. **Facebook Parent\'s Plan to Win AI Race: Give Its Tech Away Free** - Mark Zuckerberg is planning to win the AI race by giving away his company’s technology for free. This unusual strategy could potentially disrupt the AI industry. [Read more here](https://www.wsj.com/tech/ai/metas-plan-to-win-ai-race-give-its-tech-away-free-4bcc080a)\\n\\n

                2. **AI May Not Be a Job Killer, After All** - This article challenges the common notion that AI will eliminate many jobs, presenting a skeptical view on the topic. The article likely discusses evidence and arguments that contest the idea of AI as a job killer. [Read more here](https://www.wsj.com/tech/ai/will-ai-be-a-job-killer-call-us-skeptical-9b4199bd)\\n\\n

                3. **The Smartest Way to Use AI at Work** - The article provides insights on effective utilization of AI in the workplace. Popular uses for AI at work include research, brainstorming, writing first-draft emails, and creating visuals and presentations. [Read more here](https://www.wsj.com/tech/ai/the-smartest-way-to-use-ai-at-work-ce921ff4)\\n\\n

                4. **Scientists Use Generative AI to Answer Complex Questions in Physics** - Researchers have developed a new technique using generative AI to classify phase transitions in materials or physical systems. This could potentially revolutionize the field of physics. [Read more here](https://news.mit.edu/topic/artificial-intelligence2)\\n\\n

                5. **AI Tool of the Day** - Stay tuned for our AI Tool of the Day in our next update!\\n\\n

                # 💸 VC AI Investments:\\n\\n

                1. **AI & ML are Now Ubiquitous in Banks and Financial Institutions** - AI and Machine Learning are being widely adopted in the banking and financial sector, helping companies improve credit assessment and reduce fraud. This indicates the transformative impact of these technologies in the industry. [Read more here](https://economictimes.indiatimes.com/topic/artificial-intelligence-and-machine-learning)\\n\\n

                2. **Enterprises are Embracing AI for Sustainability and Growth** - AI is being adopted by enterprises to achieve sustainability and growth. It appears to offer opportunities for enterprises in terms of sustainability, growth, and navigating disruptions. [Read more here](https://economictimes.indiatimes.com/topic/articles-on-artificial-intelligence)\\n\\n

                # 📅 Upcoming Events:\\n\\n

                Check out our list of AI-related events coming up in the next week, to be shared soon!\\n\\n

                # 👋 That\'s all for this edition of AI Insights. Stay tuned for more exciting AI news!\\n\\n

                Stay curious, stay informed!\\n\\n

                The AI Newsletter Team\\n'
            """,
            callback=callback_function
        )
