import asyncio
import aiohttp
import json
import csv
import time
import pandas as pd
from typing import List, Dict, Any
import logging
import ast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MassAPITester:
    def __init__(self, csv_file_path: str, api_url: str = "http://localhost:8005/api/predict"):
        self.csv_file_path = csv_file_path
        self.api_url = api_url
        self.results = []
        
    def read_csv_data(self) -> List[str]:
        """Читает данные из CSV файла и возвращает список текстов из колонки sample"""
        samples = []
        try:
            with open(self.csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter=';')
                for row in reader:
                    if 'sample' in row:
                        samples.append(row['sample'])
            logger.info(f"Прочитано {len(samples)} образцов из файла {self.csv_file_path}")
            return samples
        except Exception as e:
            logger.error(f"Ошибка при чтении CSV файла: {e}")
            return []
    
    async def send_single_request(self, session: aiohttp.ClientSession, text: str, index: int) -> Dict[str, Any]:
        """Отправляет один запрос к API и измеряет время ответа"""
        payload = json.dumps({"input": text})
        headers = {'Content-Type': 'application/json'}
        
        start_time = time.time()
        
        try:
            async with session.post(self.api_url, headers=headers, data=payload) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                response_text = await response.text()
                
                result = {
                    'index': index,
                    'sample': text,
                    'response_time': response_time,
                    'status_code': response.status,
                    'annotation': ast.literal_eval(response_text),
                    'success': response.status == 200
                }
                
                logger.info(f"Запрос {index}: {response_time:.3f}s, статус: {response.status}")
                return result
                
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            result = {
                'index': index,
                'input_text': text,
                'response_time': response_time,
                'status_code': None,
                'response_text': str(e),
                'success': False
            }
            
            logger.error(f"Ошибка в запросе {index}: {e}")
            return result
    
    async def send_all_requests(self, samples: List[str], max_concurrent: int = 100) -> List[Dict[str, Any]]:
        """Отправляет все запросы одновременно с ограничением на количество одновременных соединений"""
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for i, sample in enumerate(samples):
                task = self.send_single_request(session, sample, i)
                tasks.append(task)
            
            logger.info(f"Отправляем {len(tasks)} запросов одновременно...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        'index': i,
                        'input_text': samples[i] if i < len(samples) else '',
                        'response_time': 0,
                        'status_code': None,
                        'response_text': str(result),
                        'success': False
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
    
    def save_results_to_csv(self, results: List[Dict[str, Any]], output_file: str = "api_test_results.csv"):
        """Сохраняет результаты в CSV файл"""
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False, encoding='utf-8', sep=';')
            logger.info(f"Результаты сохранены в файл: {output_file}")
            
            successful_requests = df[df['success'] == True]
            failed_requests = df[df['success'] == False]
            
            logger.info(f"Всего запросов: {len(df)}")
            logger.info(f"Успешных: {len(successful_requests)}")
            logger.info(f"Неудачных: {len(failed_requests)}")
            
            if len(successful_requests) > 0:
                avg_response_time = successful_requests['response_time'].mean()
                min_response_time = successful_requests['response_time'].min()
                max_response_time = successful_requests['response_time'].max()
                
                logger.info(f"Среднее время ответа: {avg_response_time:.3f}s")
                logger.info(f"Минимальное время ответа: {min_response_time:.3f}s")
                logger.info(f"Максимальное время ответа: {max_response_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов: {e}")
    
    async def run_test(self, max_concurrent: int = 100):
        logger.info("Начинаем массовое тестирование API...")

        samples = self.read_csv_data()
        if not samples:
            logger.error("Не удалось прочитать данные из CSV файла")
            return

        start_time = time.time()
        results = await self.send_all_requests(samples, max_concurrent)
        total_time = time.time() - start_time
        logger.info(f"Все запросы завершены за {total_time:.3f}s")
        self.save_results_to_csv(results)
        return results

async def main():
    csv_file = "example/sub_base.csv"

    api_url = "http://localhost:8000/api/predict"
    # api_url = "https://ai.api.vniizht.ru/api/predict"
    tester = MassAPITester(csv_file, api_url)
    await tester.run_test(max_concurrent=100)

if __name__ == "__main__":
    """
    Скрипт для массового тестирования API с измерением времени ответа.
    Читает данные из example/sub_base.csv и отправляет все запросы одновременно.
    """
    asyncio.run(main())



