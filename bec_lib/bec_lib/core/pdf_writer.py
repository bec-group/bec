import datetime

from fpdf import FPDF


class BECPDF(FPDF):
    def header(self):
        if not hasattr(self, "title"):
            return
        # Arial bold 15
        self.set_font("Courier", "", 8)
        # Calculate width of title and position
        w = self.get_string_width(self.title) + 6
        self.set_x((210 - w) / 2)
        # Title
        self.cell(w, 9, self.title, 0, 0, "C")
        # Line break
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font("Courier", "", 8)
        # Text color in gray
        self.set_text_color(128)
        # date
        self.cell(0, 10, f"BEC, {str(datetime.datetime.now())}", 0, 0, "L")
        # Page number
        self.cell(0, 10, "Page " + str(self.page_no()), 0, 0, "R")


class PDFWriter:
    LEFT_MARGIN = 25
    TOP_MARGIN = 20
    RIGHT_MARGIN = 20
    FONT_SIZE = 10

    def __init__(self, file: str, title: str = None, font="Courier") -> None:
        """Filer writer for pdfs.

        Args:
            file (str): full path to the output file (including file extension).
            title (str, optional): Title of the document. Defaults to None.

        """
        self.file = file
        self.title = title
        self.font = font
        self._pdf = BECPDF(orientation="P", unit="mm", format="A4")
        if self.title:
            self._pdf.set_title(self.title)
        self._pdf.add_page()
        self._pdf.set_margins(left=self.LEFT_MARGIN, top=self.TOP_MARGIN, right=self.RIGHT_MARGIN)
        self._pdf.set_font(self.font, "", self.FONT_SIZE)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def write(self, text: str):
        """add text to the pdf"""
        self._pdf.set_x(self.LEFT_MARGIN)
        self._pdf.multi_cell(0, self.FONT_SIZE // 2, text)

    def close(self):
        """close the pdf and write to disk"""
        self._pdf.output(self.file, "F")


if __name__ == "__main__":
    header = (
        " \n" * 3
        + "  :::            :::       :::   :::   ::::    ::: ::::::::::: \n"
        + "  :+:          :+: :+:    :+:+: :+:+:  :+:+:   :+:     :+:     \n"
        + "  +:+         +:+   +:+  +:+ +:+:+ +:+ :+:+:+  +:+     +:+     \n"
        + "  +#+        +#++:++#++: +#+  +:+  +#+ +#+ +:+ +#+     +#+     \n"
        + "  +#+        +#+     +#+ +#+       +#+ +#+  +#+#+#     +#+     \n"
        + "  #+#        #+#     #+# #+#       #+# #+#   #+#+#     #+#     \n"
        + "  ########## ###     ### ###       ### ###    #### ########### \n"
    )
    with PDFWriter("test_output.pdf") as file:
        file.write(header)
