import psycopg2


def main():
    conn = psycopg2.connect('postgres://avnadmin:<redacted>@pg-9cda73c-inseec-ce6c.e.aivencloud.com:17868/defaultdb?sslmode=require')

    query_sql = 'SELECT VERSION()'

    cur = conn.cursor()
    cur.execute(query_sql)

    version = cur.fetchone()[0]
    print(version)


if __name__ == "__main__":
    main()