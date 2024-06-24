#let details = (
  title: "Deep learning MRI lung series registration",
  date: {
    let today = datetime(
      year: 2024,
      month: 10,
      day: 1
    )
    let remainder = calc.rem(today.day(), 10)
    let suffix = "[month repr:long] [year]"

    today.display(
      if remainder == 1 {
        "[day padding:none]st "
      } else if remainder == 2 {
        "[day padding:none]nd "
      } else if remainder == 3 {
        "[day padding:none]rd "
      } else {
        "[day padding:none]th "
      } + suffix)
  },
  language: "en",
  degree: "Master",
  field: "Applied Informatics",
  fontSize: 12pt,
  doubleSided: false,
  author: (
    name: "Jamie Christopher Temple",
    role: "Autor",
    details: (
      "1717113", 
      "jamie.temple@stud.hs-hannover.de"
    ),
  ),
  examiners: (
  (
    role: "Erstprüfer",
    details: (
       "Prof. Dr. Vorname Name",
       "Abteilung Informatik, Fakultät IV",
       "Hochschule Hannover",
       "Email Adresse",
    ),
  ),
  (
    role: "Zweitprüfer",
    details: (
       "Prof. Dr. Vorname Name",
       "Abteilung Informatik, Fakultät IV",
       "Hochschule Hannover",
       "Email Adresse",
    ),
  ),
  ),
)