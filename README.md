# Introducation
This project is a FastAPI project for Abhsher hacakthon.
that detects if a given string is a valid report for online crimes or not.

It uses Python Fastapi and [Supabase database](https://supabase.com/) as a a database, and deploys on [Railway](https://railway.app/).

## Run the project : 
Sign up on a Supabase account and create a new project/ or you can use [supabase cli](https://supabase.com/docs/guides/local-development#dependencies).
with this DB schema:
```sql
CREATE TABLE IF NOT EXISTS `crimes` (
  `id` INTEGER PRIMARY KEY AUTOINCREMENT,
  `Report` TEXT NOT NULL,
  `Score` Numeric NOT NULL,
  `created_at` Date NOT NULL default current_timestamp,
  `Status` TEXT NOT NULL 
);
```

Then, Add the supabase env variable to your .env file I gave an example in .evn.example file.

- Install packages using `pip install -r requirements.txt`
- Run locally using `python main.py`


## deploy on Railway
[![Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https%3A%2F%2Fgithub.com%2Frailwayapp%2Fstarters%2Ftree%2Fmaster%2Fexamples%2Ffastapi)

### Credits
This project was made with the help of 
- [Yasir Al-Abbas](https://github.com/YasirAlabas) 
  - Designed the ml model
- [Omar Al-Abbas](https://github.com/Omer-code)
  - Designed the ml model
- [Abdullah Mzaien](https://github.com/mzaien)
  - Developed the server and deployed it 
