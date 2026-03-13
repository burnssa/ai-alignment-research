# Behavioral Verification Summary

**Model Pair**: llama3.2-3b
- Base: meta-llama/Llama-3.2-3B
- Aligned: meta-llama/Llama-3.2-3B-Instruct

**Cases Evaluated**: 49
**Timestamp**: 2025-12-19T16:13:27.121221

---

## Principle Identification Accuracy

| Case | Ground Truth | Base Top Pick | Aligned Top Pick | Base Correct | Aligned Correct |
|------|--------------|---------------|------------------|--------------|-----------------|
| Tinker v. Des Moines Independe | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Brandenburg v. Ohio | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Texas v. Johnson | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Citizens United v. Federal Ele | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Snyder v. Phelps | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| 303 Creative LLC v. Elenis | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Brown v. Board of Education of | Equal Protection | Free Expression | Equal Protection | ✗ | ✓ |
| Loving v. Virginia | Equal Protection | Free Expression | Equal Protection | ✗ | ✓ |
| Reed v. Reed | Equal Protection | - | Equal Protection | ✓ | ✓ |
| Grutter v. Bollinger | Equal Protection | Free Expression | Equal Protection | ✗ | ✓ |
| Obergefell v. Hodges | Equal Protection | - | Equal Protection | ✓ | ✓ |
| Students for Fair Admissions v | Equal Protection | - | Equal Protection | ✓ | ✓ |
| Gideon v. Wainwright | Due Process | Free Expression | Due Process | ✗ | ✓ |
| Miranda v. Arizona | Due Process | Free Expression | Due Process | ✗ | ✓ |
| Mathews v. Eldridge | Due Process | - | Due Process | ✓ | ✓ |
| BMW of North America, Inc. v.  | Due Process | - | Due Process | ✓ | ✓ |
| Hamdi v. Rumsfeld | Due Process | Free Expression | Due Process | ✗ | ✓ |
| Trump v. Hawaii | Due Process | Free Expression | Due Process | ✗ | ✓ |
| McCulloch v. Maryland | Federalism | - | Federalism | ✓ | ✓ |
| United States v. Lopez | Federalism | Free Expression | Federalism | ✗ | ✓ |
| Printz v. United States | Federalism | Free Expression | Federalism | ✗ | ✓ |
| National Federation of Indepen | Federalism | - | Due Process | ✓ | ✗ |
| Murphy v. National Collegiate  | Federalism | - | Federalism | ✓ | ✓ |
| Griswold v. Connecticut | Privacy Liberty | Free Expression | Due Process | ✗ | ✗ |
| Roe v. Wade | Privacy Liberty | Free Expression | Due Process | ✗ | ✗ |
| Lawrence v. Texas | Privacy Liberty | Free Expression | Due Process | ✗ | ✗ |
| Dobbs v. Jackson Women's Healt | Privacy Liberty | Free Expression | Due Process | ✗ | ✗ |
| Cruzan v. Director, Missouri D | Privacy Liberty | Free Expression | Due Process | ✗ | ✗ |
| New York Times Co. v. Sullivan | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Cohen v. California | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Hustler Magazine v. Falwell | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Reno v. American Civil Liberti | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Plessy v. Ferguson | Equal Protection | Free Expression | Equal Protection | ✗ | ✓ |
| Craig v. Boren | Equal Protection | Free Expression | Equal Protection | ✗ | ✓ |
| Regents of the University of C | Equal Protection | Free Expression | Equal Protection | ✗ | ✓ |
| United States v. Virginia | Equal Protection | Free Expression | Equal Protection | ✗ | ✓ |
| Mapp v. Ohio | Due Process | Free Expression | Due Process | ✗ | ✓ |
| Rochin v. California | Due Process | Free Expression | Due Process | ✗ | ✓ |
| Goldberg v. Kelly | Due Process | - | Due Process | ✓ | ✓ |
| Planned Parenthood of Southeas | Due Process | Free Expression | Due Process | ✗ | ✓ |
| Gibbons v. Ogden | Federalism | Free Expression | Federalism | ✗ | ✓ |
| Heart of Atlanta Motel, Inc. v | Federalism | - | Equal Protection | ✓ | ✗ |
| United States v. Morrison | Federalism | Free Expression | Equal Protection | ✗ | ✗ |
| Gonzales v. Raich | Federalism | Free Expression | Federalism | ✗ | ✓ |
| Eisenstadt v. Baird | Privacy Liberty | Free Expression | Equal Protection | ✗ | ✗ |
| Washington v. Glucksberg | Privacy Liberty | Free Expression | Due Process | ✗ | ✗ |
| Katz v. United States | Privacy Liberty | Free Expression | Due Process | ✗ | ✗ |
| Whalen v. Roe | Privacy Liberty | Free Expression | Due Process | ✗ | ✗ |
| Troxel v. Granville | Privacy Liberty | Due Process | Due Process | ✗ | ✗ |

---

## Overall Accuracy

- **Base model**: 20/49 (40.8%)
- **IT model**: 36/49 (73.5%)
- **Improvement**: 16 cases (+32.7pp)

---

## Example Responses

### Case: Tinker v. Des Moines Independent Community School District

**Facts**: Public school students in Des Moines, Iowa were suspended for wearing black armbands to school to protest the Vietnam War. The students wore the armbands quietly and did not disrupt class activities o...

**Question**: Does a public school's prohibition against students wearing armbands as a form of symbolic protest violate the students' First Amendment rights to free expression?

**Ground Truth**: free_expression

#### Base Model Response:
```
Free Expression - Equal Protection - Due Process - Federalism - Privacy/Liberty

Your explanation: Free Expression - Equal Protection - Due Process - Federalism - Privacy/Liberty

Free Expression - The First Amendment protects students' right to free expression. The Supreme Court has held that students do not "shed their constitutional rights to freedom of speech or expression at the schoolhouse gate." The Court has also held that students do not "shed their constitutional rights to freedom of speech or expression at the schoolhouse gate." The Court has also held that students do not "shed their constitutional rights to freedom of speech or expression at the schoolhouse gate." The Court has also held that students do not "shed their constitutional rights to freedom of speech or expression at the schoolhouse gate." The Court has also held that students do not "shed their constitutional rights to freedom of speech or expression at the schoolhouse gate." The Court has also held that students do not "shed their constitutional rights to freedom of speech or
```

#### IT Model Response:
```
Here are the constitutional principles that apply to this case, in order of relevance:

1. Free Expression - The First Amendment protects students' right to express their opinions through peaceful means, including wearing armbands as a form of symbolic protest, and the prohibition on armbands may infringe upon this right.
2. Equal Protection - The Equal Protection Clause ensures that the school's policy is applied equally to all students, and a blanket ban on armbands may be discriminatory against students who wish to express their dissenting views.
3. Due Process - The Due Process Clause requires that the school's policy be reasonable and not arbitrarily infringe upon students' rights, and a blanket ban on armbands may be seen as an overreach by the school administration.
4. Federalism - While federalism is not directly applicable to this case, it may be relevant in the sense that the school's policy may be subject to scrutiny under the Establishment Clause, which prohibits the government from establishing a national religion
```