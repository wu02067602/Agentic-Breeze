import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any, Tuple, Union
import os

# 定義 SQLite 支援的資料類型
SQLiteType = Literal["TEXT", "INTEGER", "REAL", "BLOB", "NUMERIC"]

@dataclass
class ColumnDefinition:
    """定義資料表中的一個欄位。"""
    name: str
    dtype: SQLiteType
    nullable: bool = True
    is_primary_key: bool = False
    is_unique: bool = False
    default: Optional[str] = None
    description: Optional[str] = None

@dataclass
class TableDefinition:
    """定義一個資料表的 schema。"""
    name: str
    columns: List[ColumnDefinition]
    description: Optional[str] = None
    # 未來可以考慮添加 indexes: List[IndexDefinition] 等更複雜的定義


class SQLiteSchemaTool:
    """
    一個用於管理 SQLite 資料庫 schema 和執行查詢的工具。
    """
    def __init__(self, db_path: str = "sample_users.db"):
        """
        初始化 SQLite Schema 工具。
        
        Args:
            db_path (str): 資料庫檔案路徑，預設為 "sample_users.db"
        
        Returns:
            None
        
        Examples:
            >>> tool = SQLiteSchemaTool(db_path="my_database.db")
            >>> isinstance(tool, SQLiteSchemaTool)
            True
        
        Raises:
            無特定錯誤
        """
        self._db_path = db_path

    def _table_exists(self, cursor: sqlite3.Cursor, table_name: str) -> bool:
        """
        檢查資料表是否存在。
        
        Args:
            cursor (sqlite3.Cursor): SQLite 游標物件
            table_name (str): 要檢查的資料表名稱
        
        Returns:
            bool: 若資料表存在則為 True，否則為 False
        
        Examples:
            >>> tool = SQLiteSchemaTool()
            >>> with sqlite3.connect(tool._db_path) as conn:
            ...     cursor = conn.cursor()
            ...     exists = tool._table_exists(cursor, "users")
        
        Raises:
            sqlite3.Error: 當資料庫查詢發生錯誤時
        """
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        return cursor.fetchone() is not None

    def _get_existing_columns(self, cursor: sqlite3.Cursor, table_name: str) -> Dict[str, Dict[str, Any]]:
        """
        獲取現有資料表的欄位資訊。
        
        Args:
            cursor (sqlite3.Cursor): SQLite 游標物件
            table_name (str): 資料表名稱
        
        Returns:
            Dict[str, Dict[str, Any]]: 欄位名稱對應的欄位資訊字典
        
        Examples:
            >>> tool = SQLiteSchemaTool()
            >>> with sqlite3.connect(tool._db_path) as conn:
            ...     cursor = conn.cursor()
            ...     columns = tool._get_existing_columns(cursor, "users")
        
        Raises:
            sqlite3.Error: 當資料庫查詢發生錯誤時
        """
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns_info = {}
        for row in cursor.fetchall():
            columns_info[row['name']] = {
                "name": row['name'],
                "type": row['type'],
                "notnull": bool(row['notnull']),
                "pk": bool(row['pk']),
                "dflt_value": row['dflt_value']
            }
        return columns_info

    def _generate_create_table_ddl(self, table_definition: TableDefinition) -> str:
        """
        根據 TableDefinition 生成 CREATE TABLE 語句。
        
        Args:
            table_definition (TableDefinition): 資料表定義物件
        
        Returns:
            str: CREATE TABLE SQL 語句
        
        Examples:
            >>> from agentic_breeze.agents.orchestrator_core.tools.sqlite_tool import TableDefinition, ColumnDefinition
            >>> col = ColumnDefinition(name="id", dtype="INTEGER", is_primary_key=True)
            >>> table = TableDefinition(name="users", columns=[col])
            >>> tool = SQLiteSchemaTool()
            >>> sql = tool._generate_create_table_ddl(table)
        
        Raises:
            無特定錯誤
        """
        columns_sql = []
        primary_keys = []
        for col in table_definition.columns:
            col_sql = f"{col.name} {col.dtype}"
            if not col.nullable:
                col_sql += " NOT NULL"
            if col.is_unique:
                col_sql += " UNIQUE"
            if col.default is not None:
                # 確保預設值格式正確，例如字串需要單引號
                if col.dtype == "TEXT":
                    col_sql += f" DEFAULT '{col.default}'"
                else:
                    col_sql += f" DEFAULT {col.default}"
            if col.is_primary_key:
                primary_keys.append(col.name)
            columns_sql.append(col_sql)

        if primary_keys:
            columns_sql.append(f"PRIMARY KEY ({', '.join(primary_keys)})")

        return f"CREATE TABLE {table_definition.name} ({', '.join(columns_sql)})"

    def _generate_migrate_ddl(self, table_definition: TableDefinition,
                              existing_columns: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        比對 schema 並生成非破壞性 ALTER TABLE 語句（追加欄位和索引）。
        
        Args:
            table_definition (TableDefinition): 資料表定義物件
            existing_columns (Dict[str, Dict[str, Any]]): 現有欄位資訊字典
        
        Returns:
            List[str]: ALTER TABLE SQL 語句列表
        
        Examples:
            >>> from agentic_breeze.agents.orchestrator_core.tools.sqlite_tool import TableDefinition, ColumnDefinition
            >>> col = ColumnDefinition(name="email", dtype="TEXT")
            >>> table = TableDefinition(name="users", columns=[col])
            >>> tool = SQLiteSchemaTool()
            >>> existing = {}
            >>> ddl = tool._generate_migrate_ddl(table, existing)
        
        Raises:
            無特定錯誤
        """
        ddl_statements = []
        for new_col in table_definition.columns:
            if new_col.name not in existing_columns:
                # 追加新欄位
                col_sql = f"{new_col.name} {new_col.dtype}"
                if not new_col.nullable:
                    col_sql += " NOT NULL"
                if new_col.is_unique:
                    col_sql += " UNIQUE"
                if new_col.default is not None:
                    if new_col.dtype == "TEXT":
                        col_sql += f" DEFAULT '{new_col.default}'"
                    else:
                        col_sql += f" DEFAULT {new_col.default}"
                ddl_statements.append(f"ALTER TABLE {table_definition.name} ADD COLUMN {col_sql}")
            # 簡化處理：對於現有欄位，不允許更改類型或刪除，避免數據破壞

        # 這裡可以添加索引的處理邏輯，例如為 is_unique 和 is_primary_key 的欄位自動創建索引
        # 或者支援 TableDefinition 中明確定義的索引
        return ddl_statements

    def define_table(self, table_definition: TableDefinition) -> Dict[str, Any]:
        """
        根據 dataclass 定義的 schema 遷移資料表。
        支持非破壞性遷移（追加欄位）。不允許創建新表。
        
        Args:
            table_definition (TableDefinition): 資料表定義物件
        
        Returns:
            Dict[str, Any]: 包含 status 和 message 的結果字典
        
        Examples:
            >>> from agentic_breeze.agents.orchestrator_core.tools.sqlite_tool import TableDefinition, ColumnDefinition
            >>> col = ColumnDefinition(name="age", dtype="INTEGER")
            >>> table = TableDefinition(name="users", columns=[col])
            >>> tool = SQLiteSchemaTool()
            >>> result = tool.define_table(table)
            >>> result["status"] in ["success", "error"]
            True
        
        Raises:
            sqlite3.Error: 當資料庫操作發生錯誤時
        """
        try:
            # 在嘗試獲取連接之前，先檢查資料庫檔案是否存在。
            # 如果檔案不存在，則直接返回錯誤，因為設定LLM 無法創建新的資料庫檔案。
            if not os.path.exists(self._db_path):
                return {"status": "error", "message": f"Database file '{self._db_path}' does not exist. New table creation is not allowed.", "error_details": "Database file not found and creation disallowed"}

            with sqlite3.connect(self._db_path, timeout=30) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                table_name = table_definition.name

                if not self._table_exists(cursor, table_name):
                    return {"status": "error", "message": f"Table '{table_name}' does not exist. New table creation is not allowed.", "error_details": "Table not found and creation disallowed"}
                else:
                    # 資料表已存在，比對並遷移
                    existing_columns = self._get_existing_columns(cursor, table_name)
                    ddl_statements = self._generate_migrate_ddl(table_definition, existing_columns)
                    if ddl_statements:
                        for ddl in ddl_statements:
                            cursor.execute(ddl)
                        conn.commit()
                        return {"status": "success", "message": f"Table '{table_name}' migrated successfully. Added {len(ddl_statements)} new columns."}
                    else:
                        return {"status": "success", "message": f"Table '{table_name}' schema is up to date."}

        except sqlite3.Error as e:
            # with 語句會自動處理連接關閉，這裡只需處理回滾
            return {"status": "error", "message": f"Failed to define/migrate table '{table_name}': {e}", "error_details": str(e)}

    def query_to_dicts(self, sql_query: str, params: Optional[Union[Tuple, Dict]] = None) -> Dict[str, Any]:
        """
        執行 SELECT 語句並回傳結果為字典列表。
        
        Args:
            sql_query (str): SQL 查詢語句
            params (Optional[Union[Tuple, Dict]]): 查詢參數，可為 tuple 或 dict，預設為 None
        
        Returns:
            Dict[str, Any]: 包含 status、data 和 row_count 的結果字典
        
        Examples:
            >>> tool = SQLiteSchemaTool()
            >>> result = tool.query_to_dicts("SELECT * FROM users WHERE id = ?", (1,))
            >>> result["status"]
            'success'
        
        Raises:
            sqlite3.Error: 當 SQL 查詢執行發生錯誤時
        """
        try:
            with sqlite3.connect(self._db_path, timeout=30) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if params:
                    cursor.execute(sql_query, params)
                else:
                    cursor.execute(sql_query)
                
                rows = cursor.fetchall()
                # sqlite3.Row 已經讓結果可以像字典一樣訪問
                results = [dict(row) for row in rows]
                
                return {"status": "success", "data": results, "row_count": len(results)}
        except sqlite3.Error as e:
            return {"status": "error", "message": f"Failed to query data: {e}", "error_details": str(e)}

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        獲取指定資料表的詳細資訊，包括欄位、型別等。
        
        Args:
            table_name (str): 資料表名稱
        
        Returns:
            Dict[str, Any]: 包含 status、table_name 和 columns 的結果字典
        
        Examples:
            >>> tool = SQLiteSchemaTool()
            >>> result = tool.get_table_info("users")
            >>> result["status"] in ["success", "error"]
            True
        
        Raises:
            sqlite3.Error: 當資料庫查詢發生錯誤時
        """
        try:
            with sqlite3.connect(self._db_path, timeout=30) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if not self._table_exists(cursor, table_name):
                    return {"status": "error", "message": f"Table '{table_name}' does not exist.", "error_details": "Table not found"}

                columns_info = self._get_existing_columns(cursor, table_name)
                
                return {"status": "success", "table_name": table_name, "columns": columns_info}
        except sqlite3.Error as e:
            return {"status": "error", "message": f"Failed to get table info: {e}", "error_details": str(e)}