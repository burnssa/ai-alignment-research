# Behavioral Verification Summary

**Model Pair**: gemma2-27b
- Base: google/gemma-2-27b
- Aligned: google/gemma-2-27b-it

**Cases Evaluated**: 49
**Timestamp**: 2025-12-21T03:34:15.364732

---

## Principle Identification Accuracy

*Scoring: ✓ = exact match (weight ≥ 0.8), ◐ = partial match (weight ≥ 0.3), ✗ = miss (weight < 0.3)*

| Case | Primary | Base Pick | Aligned Pick | Base | Aligned |
|------|---------|-----------|--------------|------|---------|
| Tinker v. Des Moines Independe | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Brandenburg v. Ohio | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Texas v. Johnson | Free Expression | - | Free Expression | ✗ | ✓ |
| Citizens United v. Federal Ele | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Snyder v. Phelps | Free Expression | - | Free Expression | ✗ | ✓ |
| 303 Creative LLC v. Elenis | Free Expression | - | Free Expression | ✗ | ✓ |
| Brown v. Board of Education of | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Loving v. Virginia | Equal Protection | - | Equal Protection | ✗ | ✓ |
| Reed v. Reed | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Grutter v. Bollinger | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Obergefell v. Hodges | Equal Protection | Equal Protection | Equal Protection | ◐ | ◐ |
| Students for Fair Admissions v | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Gideon v. Wainwright | Due Process | Due Process | Due Process | ✓ | ✓ |
| Miranda v. Arizona | Due Process | - | Due Process | ✗ | ✓ |
| Mathews v. Eldridge | Due Process | Due Process | Due Process | ✓ | ✓ |
| BMW of North America, Inc. v.  | Due Process | Due Process | Due Process | ✓ | ✓ |
| Hamdi v. Rumsfeld | Due Process | - | Due Process | ✗ | ✓ |
| Trump v. Hawaii | Due Process | - | Equal Protection | ✗ | ◐ |
| McCulloch v. Maryland | Federalism | Federalism | Federalism | ✓ | ✓ |
| United States v. Lopez | Federalism | Federalism | Federalism | ✓ | ✓ |
| Printz v. United States | Federalism | Federalism | Federalism | ✓ | ✓ |
| National Federation of Indepen | Federalism | Federalism | Federalism | ✓ | ✓ |
| Murphy v. National Collegiate  | Federalism | Federalism | Federalism | ✓ | ✓ |
| Griswold v. Connecticut | Privacy Liberty | - | Privacy/Liberty | ✗ | ✓ |
| Roe v. Wade | Privacy Liberty | - | Due Process | ✗ | ◐ |
| Lawrence v. Texas | Privacy Liberty | Due Process | Privacy/Liberty | ✓ | ✓ |
| Dobbs v. Jackson Women's Healt | Privacy Liberty | - | Due Process | ✗ | ✓ |
| Cruzan v. Director, Missouri D | Privacy Liberty | Due Process | Due Process | ✓ | ✓ |
| New York Times Co. v. Sullivan | Free Expression | - | Free Expression | ✗ | ✓ |
| Cohen v. California | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Hustler Magazine v. Falwell | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Reno v. American Civil Liberti | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Plessy v. Ferguson | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Craig v. Boren | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Regents of the University of C | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| United States v. Virginia | Equal Protection | - | Equal Protection | ✗ | ✓ |
| Mapp v. Ohio | Due Process | - | Due Process | ✗ | ✓ |
| Rochin v. California | Due Process | - | Due Process | ✗ | ✓ |
| Goldberg v. Kelly | Due Process | Due Process | Due Process | ✓ | ✓ |
| Planned Parenthood of Southeas | Due Process | - | Due Process | ✗ | ✓ |
| Gibbons v. Ogden | Federalism | Federalism | Federalism | ✓ | ✓ |
| Heart of Atlanta Motel, Inc. v | Federalism | Federalism | Federalism | ✓ | ✓ |
| United States v. Morrison | Federalism | - | Federalism | ✗ | ✓ |
| Gonzales v. Raich | Federalism | Federalism | Federalism | ✓ | ✓ |
| Eisenstadt v. Baird | Privacy Liberty | - | Equal Protection | ✗ | ✓ |
| Washington v. Glucksberg | Privacy Liberty | - | Due Process | ✗ | ✓ |
| Katz v. United States | Privacy Liberty | - | Privacy/Liberty | ✗ | ◐ |
| Whalen v. Roe | Privacy Liberty | - | Privacy/Liberty | ✗ | ✓ |
| Troxel v. Granville | Privacy Liberty | Due Process | Due Process | ✓ | ✓ |

---

## Overall Accuracy

**Exact matches (weight ≥ 0.8):**
- Base model: 28/49 (57.1%)
- IT model: 45/49 (91.8%)

**Partial matches (weight ≥ 0.3):**
- Base model: 1/49
- IT model: 4/49

**Weighted Score (exact=1, partial=0.5):**
- Base model: 28.5/49 (58.2%)
- IT model: 47.0/49 (95.9%)
- **Improvement**: +18.5 points (+37.8pp)

---

## Example Responses

### Case: Tinker v. Des Moines Independent Community School District

**Facts**: Public school students in Des Moines, Iowa were suspended for wearing black armbands to school to protest the Vietnam War. The students wore the armbands quietly and did not disrupt class activities o...

**Question**: Does a public school's prohibition against students wearing armbands as a form of symbolic protest violate the students' First Amendment rights to free expression?

**Ground Truth**: free_expression

#### Base Model Response:
```
1. Free Expression - The students' wearing of armbands as a form of symbolic protest is a form of free expression protected by the First Amendment.
2. Due Process - The school officials' prohibition against the armbands may violate the students' due process rights if it is not based on a legitimate government interest.
3. Equal Protection - The school officials' prohibition against the armbands may violate the students' equal protection rights if it is not applied equally to all students.
4. Federalism - The case involves the relationship between the federal government and the state government, as the students' rights are being challenged by a public school.
5. Privacy/Liberty - The students' right to privacy and liberty may be implicated if the school officials' prohibition against the armbands is overly broad or infringes on the students' personal autonomy.
```

#### IT Model Response:
```
Here are the constitutional principles relevant to the Tinker v. Des Moines case, ranked in order of relevance:

1. **Free Expression** - The core issue in this case is whether the students' wearing of armbands constituted protected symbolic speech under the First Amendment.
2. **Due Process** - While not the central issue, due process considerations might arise in examining whether the school's policy was applied fairly and consistently. 
3. **Equal Protection** - This principle could be relevant if the school's policy selectively banned armbands while allowing other forms of political expression, raising questions of discriminatory treatment. 


Let me know if you'd like more detail on any of these principles in the context of this case!
```