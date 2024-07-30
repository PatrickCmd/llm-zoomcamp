# PostgreSQL Shell Commands

## Connect to database using pgcli

Certainly! You can use pgcli to connect to a containerized PostgreSQL database from outside the container. Here's how you can do it:

1. First, make sure you have pgcli installed. If not, you can install it using pip:

   ```
   pip install pgcli
   ```

2. Assuming your PostgreSQL container is running and exposing its port (as we discussed in the previous answer), you can use pgcli with the following command structure:

   ```
   pgcli -h localhost -p 5432 -U your_username -d your_database_name
   ```

Here's a breakdown of the parameters:

- `-h localhost`: This specifies the host. Use 'localhost' if the container is on the same machine.
- `-p 5432`: This is the port number. 5432 is the default for PostgreSQL, but use whatever port you've exposed from your container.
- `-U your_username`: Replace this with your PostgreSQL username.
- `-d your_database_name`: Replace this with the name of your database.

For example, if your database is named 'mydb', your username is 'postgres', and you're using the default port, your command would look like this:

```
pgcli -h localhost -p 5432 -U postgres -d mydb
```

After entering this command, you'll be prompted for the password.

If you want to connect to the default 'postgres' database instead of a specific database, you can omit the `-d` option:

```
pgcli -h localhost -p 5432 -U postgres
```

```
pgcli -h localhost -p 5433 -U postgres -d course_assistant -W
```

Remember:
1. If your PostgreSQL container is running on a different machine, replace 'localhost' with that machine's IP address.
2. If you've mapped the PostgreSQL port to a different port on the host, use that port number instead of 5432.
3. Make sure your PostgreSQL container is configured to allow connections from external hosts.

Once connected, you can interact with your PostgreSQL database using SQL commands, just as you would in any other PostgreSQL client. pgcli offers features like auto-completion and syntax highlighting, which can make your database interactions more efficient.

## Connection Commands

- `\c dbname` - Connect to a different database
- `\conninfo` - Display current connection information

## Help and Information

### General Help
- `\h` - Get help on SQL commands
- `\?` - Get help on psql commands

### Database Objects
- `\d` - List tables, views, and sequences
- `\d tablename` - Describe a table's structure
- `\df` - List functions
- `\du` - List roles (users)
- `\l` - List databases

## Query Execution

- Enter SQL queries directly and end with a semicolon (;)
- `\g` - Execute the last command again
- `\e` - Edit the last command in an external editor

## Output Control

- `\x` - Toggle expanded table formatting
- `\a` - Toggle between aligned and unaligned output mode
- `\H` - Toggle HTML output format
- `\t` - Show only rows (no headers)

## Transaction Control

- `BEGIN` - Start a transaction
- `COMMIT` - Commit the current transaction
- `ROLLBACK` - Roll back the current transaction

## Database Operations

- `CREATE DATABASE dbname;` - Create a new database
- `DROP DATABASE dbname;` - Delete a database

## Table Operations

- `CREATE TABLE tablename (column1 datatype, column2 datatype, ...);` - Create a new table
- `DROP TABLE tablename;` - Delete a table
- `ALTER TABLE tablename ADD COLUMN columnname datatype;` - Add a new column to a table

## Data Manipulation

### Inserting Data
- `INSERT INTO tablename (column1, column2, ...) VALUES (value1, value2, ...);`

### Querying Data
- `SELECT * FROM tablename;` - Select all data from a table

### Updating Data
- `UPDATE tablename SET column1 = value1 WHERE condition;`

### Deleting Data
- `DELETE FROM tablename WHERE condition;`

## User Management

- `CREATE USER username WITH PASSWORD 'password';` - Create a new user
- `ALTER USER username WITH PASSWORD 'newpassword';` - Change a user's password
- `GRANT ALL PRIVILEGES ON DATABASE dbname TO username;` - Grant privileges

## Exiting

- `\q` - Quit psql

## Utility Commands

- `\timing` - Toggle timing of commands
- `\i filename` - Execute commands from a file
- `\o filename` - Send query results to a file

---

**Note:** SQL commands (like SELECT, INSERT, CREATE TABLE, etc.) need to be terminated with a semicolon (;), while psql meta-commands (starting with \) do not.