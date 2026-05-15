import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def render_html_table(
    df: pd.DataFrame,
    title: str | None = None,
    max_height: int = 350
):
    if df is None or df.empty:
        st.info("No data to display.")
        return

    if title:
        st.markdown(f"### {title}")

    # ---------- AUTO HEIGHT (Adjusted for Bigger Rows) ----------
    ROW_HEIGHT = 40
    HEADER_HEIGHT = 60
    PADDING = 30

    rows = len(df)
    calculated_height = HEADER_HEIGHT + (rows * ROW_HEIGHT) + PADDING
    final_height = min(calculated_height, max_height)

    table_html = df.to_html(
        index=False,
        classes="display",
        table_id="formalTable",
        escape=False
    )

    html = f"""
    <html>
    <head>
        <link rel="stylesheet"
              href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">

        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

        <style>

            :root {{
                --header-bg: #1F3A5F;
                --header-text: #FFFFFF;
                --row-odd: #E6F0FF;
                --row-even: #F8FAFC;
                --row-hover: #D3E2F5;
                --row-divider: #EEF2F7;
            }}

            body {{
                margin: 0;
                padding: 0;
                font-family: 'Inter', sans-serif;
                background: transparent;
            }}

            .table-container {{
                max-height: {final_height}px;
                overflow: hidden; 
                border-radius: 18px;
                border: 1px solid #E5E7EB;
                box-shadow: 0 6px 18px rgba(0,0,0,0.06);
                background: white;
            }}

            .dataTables_scroll {{
    border-radius: 18px !important;
    overflow: hidden !important;
}}

.dataTables_scrollBody {{
    border-radius: 0 0 18px 18px !important;
    overflow: auto !important;
}}
.dataTables_scrollHead {{
    border-radius: 18px 18px 0 0 !important;
    overflow: hidden !important;
}}


            table.dataTable {{
                width: 100% !important;
                font-size: 13px;                /* Bigger font */
                border-collapse: collapse;
                border: none !important;
            }}

            /* Remove sorted column highlight */
            table.dataTable.display tbody tr > .sorting_1,
            table.dataTable.display tbody tr > .sorting_2,
            table.dataTable.display tbody tr > .sorting_3 {{
                background: inherit !important;
            }}

            /* HEADER */
            table.dataTable thead th {{
                position: sticky;
                top: 0;
                background: var(--header-bg);
                color: var(--header-text);
                font-weight: 600;
                text-align: center !important;
                padding: 16px 14px;            /* Bigger header padding */
                font-size: 13px;
                white-space: nowrap;
                letter-spacing: 0.3px;
            }}

            /* BODY CELLS */
            table.dataTable td {{
                padding: 14px 14px;            /* Bigger rows */
                border-bottom: 1px solid var(--row-divider);
                text-align: center !important;
                white-space: nowrap;
            }}

            /* Zebra */
            table.dataTable tbody tr:nth-child(odd) {{
                background-color: var(--row-odd);
            }}

            table.dataTable tbody tr:nth-child(even) {{
                background-color: var(--row-even);
            }}

            table.dataTable tbody tr:hover {{
                background-color: var(--row-hover);
                transition: background 0.2s ease;
            }}

            table.dataTable th,
            table.dataTable td {{
                border-left: none !important;
                border-right: none !important;
            }}

            .dataTables_filter,
            .dataTables_length,
            .dataTables_info {{
                display: none;
            }}

            .dataTables_wrapper {{
                margin: 0 !important;
                padding: 0 !important;
                border-radius: 18px !important;  
                overflow: hidden !important; 
            }}
            

        </style>
    </head>

    <body>
        <div class="table-container">
            {table_html}
        </div>

        <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>

        <script>
            $('#formalTable').DataTable({{
    paging: false,
    ordering: true,
    searching: false,
    info: false,
    scrollX: true,
    scrollY: "{final_height - 60}px",
    scrollCollapse: true
}});

        </script>
    </body>
    </html>
    """

    components.html(
        html,
        height=final_height,
        scrolling=False
    )