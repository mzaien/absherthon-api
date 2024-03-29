import os
from supabase import create_client, Client
url = os.getenv("SUPABASE_URL", "https://csduwsbnhkcdxalrfcem.supabase.co")
key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNzZHV3c2JuaGtjZHhhbHJmY2VtIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NDc3MDMxMDQsImV4cCI6MTk2MzI3OTEwNH0.h5EjQPuuHU4O30d_2EoU9_8E7l4CD9OnTWYM2DnUmQ8")
supabase: Client = create_client(url, key)


def list():
    all = supabase.table('Reports').select('*', count='exact').execute()
    query = supabase.table('Reports').select(
        '*', count='exact').eq('Report', "ابو عنتر هددني في تويتر").execute()
    response = {
        "all": all,
        "query": query.count
    }
    return response


def insert(data):
    supabase.table('Reports').insert(
        {'Report': data["criem_report"], 'Score': data["score"], 'Status': data["type"]}).execute()
