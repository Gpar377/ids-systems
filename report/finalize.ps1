# Open the generated report in Word, fill in the table of contents (Word must lay
# out the pages to know the numbers), save, and export a PDF copy for checking.
#   powershell -ExecutionPolicy Bypass -File report\finalize.ps1
param([string]$Docx = (Join-Path $PSScriptRoot "IDS_Case_Study_Report.docx"))

$Docx = (Resolve-Path $Docx).Path
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx)
    # update twice: filling the TOC can push later headings onto new pages
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $null = $doc.Fields.Update()
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Save()
    $pdf = [IO.Path]::ChangeExtension($Docx, ".pdf")
    $doc.ExportAsFixedFormat($pdf, 17)     # 17 = wdExportFormatPDF
    "pages: " + $doc.ComputeStatistics(2)  # 2 = wdStatisticPages
    $doc.Close([ref]$false)
    "saved $Docx"
    "pdf   $pdf"
} finally {
    $word.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
