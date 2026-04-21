// function expandableTextRenderer(params) {
//   const maxLen = 80;
//   const fullText = params.value == null ? "" : String(params.value);
//   const isLong = fullText.length > maxLen;

//   let expanded = false;

//   const eGui = document.createElement("div");
//   eGui.style.whiteSpace = "normal";
//   eGui.style.lineHeight = "1.3";

//   const textSpan = document.createElement("span");
//   const toggle = document.createElement("button");
//   toggle.type = "button";
//   toggle.style.marginLeft = "6px";
//   toggle.style.border = "none";
//   toggle.style.background = "none";
//   toggle.style.padding = "0";
//   toggle.style.cursor = "pointer";
//   toggle.style.color = "#0b5fff";
//   toggle.style.fontSize = "12px";

//   function render() {
//     textSpan.textContent =
//       !isLong || expanded ? fullText : fullText.slice(0, maxLen) + "...";

//     toggle.textContent = expanded ? "show less" : "show more";
//     toggle.style.display = isLong ? "inline" : "none";
//   }

//   toggle.addEventListener("click", (e) => {
//     e.preventDefault();
//     e.stopPropagation();
//     expanded = !expanded;
//     render();

//     if (params.api) {
//       params.api.resetRowHeights();
//     }
//   });

//   eGui.appendChild(textSpan);
//   eGui.appendChild(toggle);
//   render();

//   return eGui;
// }

// let gridApi = null;

// function initGrid() {
//   const root = document.querySelector("#table-root");
//   if (!root || root.dataset.agReady === "true") return;
//   if (typeof agGrid === "undefined") return;

//   const rowTag = document.querySelector("#table-root-data");
//   const colTag = document.querySelector("#table-root-cols");
//   const searchInput = document.querySelector("#table-search");

//   if (!rowTag || !colTag) return;

//   let rowData = [];
//   let colDefs = [];

//   try {
//     rowData = JSON.parse(rowTag.textContent || "[]");
//     colDefs = JSON.parse(colTag.textContent || "[]");
//   } catch (e) {
//     console.error("JSON parse error:", e);
//     console.log("Row raw:", rowTag.textContent);
//     console.log("Col raw:", colTag.textContent);
//     return;
//   }

//   const INDEX_FIELDS = ["index", "level_0"];

//   rowData = rowData.map(row => {
//     const cleaned = { ...row };
//     INDEX_FIELDS.forEach(f => delete cleaned[f]);
//     return cleaned;
//   });

//   const LONG_TEXT_COLUMNS = ["evidence", "a_to_b_mapping", "classification"];

//   colDefs = colDefs
//   .filter(col => !INDEX_FIELDS.includes(col.field))
//   .map(col => {
//     const isLongText = LONG_TEXT_COLUMNS.includes(col.field);

//     return {
//       ...col,
//       wrapText: true,
//       autoHeight: true,
//       cellRenderer: isLongText ? expandableTextRenderer : undefined,
//     };
//   });

//   const gridOptions = {
//     columnDefs: colDefs,
//     rowData: rowData,

//     defaultColDef: {
//       sortable: true,
//       filter: true,
//       resizable: true,
//       minWidth: 120,
//     },

//     pagination: true,
//     paginationPageSize: 25,

//     onGridReady: (params) => {
//       gridApi = params.api;

//       const allCols = params.columnApi.getColumns().map(c => c.getId());
//       params.columnApi.autoSizeColumns(allCols);

//       allCols.forEach(colId => {
//         const col = params.columnApi.getColumn(colId);
//         const width = col.getActualWidth();
//         if (width > 400) {
//           params.columnApi.setColumnWidth(colId, 400);
//         }
//       });
//     }
//   };

//   agGrid.createGrid(root, gridOptions);

//   // Hook up search box
//   if (searchInput) {
//     searchInput.addEventListener("input", (e) => {
//       const value = e.target.value || "";
//       if (gridApi) {
//         gridApi.setGridOption("quickFilterText", value);
//       }
//     });
//   }

//   root.dataset.agReady = "true";
// }

// window.addEventListener("load", initGrid);
// new MutationObserver(initGrid).observe(document.body, { childList: true, subtree: true });
function toggleExpand(el) {
  if (el.classList.contains("expandable-cell")) {
      el.classList.remove("expandable-cell");
  } else {
      el.classList.add("expandable-cell");
  }
}

function parseCellValue(text) {
  const t = (text || '').trim().replace(/,/g, '');
  if (!t) return { type: 0, value: '' };

  const num = Number(t);
  if (!Number.isNaN(num) && /^-?\d+(\.\d+)?$/.test(t)) {
      return { type: 1, value: num };
  }

  const dt = Date.parse(t);
  if (!Number.isNaN(dt)) {
      return { type: 2, value: dt };
  }

  return { type: 3, value: t.toLowerCase() };
}

function sortHtmlTable(tableId, colIndex) {
  const table = document.getElementById(tableId);
  if (!table) return;

  const tbody = table.tBodies[0];
  if (!tbody) return;

  const currentCol = table.dataset.sortCol;
  const currentDir = table.dataset.sortDir || 'asc';
  const asc = !(currentCol === String(colIndex) && currentDir === 'asc');

  const rows = Array.from(tbody.rows);
  rows.sort((r1, r2) => {
      const a = parseCellValue(r1.cells[colIndex]?.innerText || '');
      const b = parseCellValue(r2.cells[colIndex]?.innerText || '');

      if (a.type !== b.type) return a.type - b.type;

      let cmp = 0;
      if (a.value < b.value) cmp = -1;
      else if (a.value > b.value) cmp = 1;

      return asc ? cmp : -cmp;
  });

  rows.forEach(r => tbody.appendChild(r));
  table.dataset.sortCol = String(colIndex);
  table.dataset.sortDir = asc ? 'asc' : 'desc';

  const indicators = table.querySelectorAll('.sort-indicator');
  indicators.forEach(ind => ind.textContent = '');

  const th = table.querySelectorAll('th')[colIndex];
  if (th) {
      const ind = th.querySelector('.sort-indicator');
      if (ind) ind.textContent = asc ? ' ▲' : ' ▼';
  }
}