from resp.resp.resp import Resp
from paperFilter import paperFilter
from discussion import Discussion

api_key = "f6844e3f92c08e99c541d280ce5f7a6dcc1d9e505b6715ac2b42cc2850488f74"

paper_engine = Resp(api_key)

website_result = paper_engine.custom_search(url       = 'https://link.springer.com', 
                                           keyword   = 'Zero-shot learning', 
                                           max_pages = 1)

paper_filter = paperFilter(model = "mlx-community/Meta-Llama-3-8B-Instruct-4bit")

filtered_papers = paper_filter.filter_papers(website_result)

discussion = Discussion()

discussion.discuss(filtered_papers)

