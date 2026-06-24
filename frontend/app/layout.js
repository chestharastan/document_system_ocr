import "./globals.css";

export const metadata = {
  title: "Khmer Document OCR",
  description:
    "Layout analysis, text detection and recognition for Khmer documents, with an editable rebuilt page.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
