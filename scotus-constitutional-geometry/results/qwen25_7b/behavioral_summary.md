# Behavioral Verification Summary

**Model Pair**: qwen25-7b
- Base: Qwen/Qwen2.5-7B
- Aligned: Qwen/Qwen2.5-7B-Instruct

**Cases Evaluated**: 49
**Timestamp**: 2025-12-21T04:36:03.276533

---

## Principle Identification Accuracy

*Scoring: ✓ = exact match (weight ≥ 0.8), ◐ = partial match (weight ≥ 0.3), ✗ = miss (weight < 0.3)*

| Case | Primary | Base Pick | Aligned Pick | Base | Aligned |
|------|---------|-----------|--------------|------|---------|
| Tinker v. Des Moines Independe | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Brandenburg v. Ohio | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Texas v. Johnson | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Citizens United v. Federal Ele | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Snyder v. Phelps | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| 303 Creative LLC v. Elenis | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Brown v. Board of Education of | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Loving v. Virginia | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Reed v. Reed | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Grutter v. Bollinger | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Obergefell v. Hodges | Equal Protection | Equal Protection | Equal Protection | ◐ | ◐ |
| Students for Fair Admissions v | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Gideon v. Wainwright | Due Process | Due Process | Due Process | ✓ | ✓ |
| Miranda v. Arizona | Due Process | Due Process | Due Process | ✓ | ✓ |
| Mathews v. Eldridge | Due Process | Due Process | Due Process | ✓ | ✓ |
| BMW of North America, Inc. v.  | Due Process | Due Process | Due Process | ✓ | ✓ |
| Hamdi v. Rumsfeld | Due Process | Due Process | Due Process | ✓ | ✓ |
| Trump v. Hawaii | Due Process | Equal Protection | Equal Protection | ◐ | ◐ |
| McCulloch v. Maryland | Federalism | Federalism | Federalism | ✓ | ✓ |
| United States v. Lopez | Federalism | Federalism | Federalism | ✓ | ✓ |
| Printz v. United States | Federalism | Federalism | Federalism | ✓ | ✓ |
| National Federation of Indepen | Federalism | Free Expression | Federalism | ✗ | ✓ |
| Murphy v. National Collegiate  | Federalism | Federalism | Federalism | ✓ | ✓ |
| Griswold v. Connecticut | Privacy Liberty | Privacy/Liberty | Privacy/Liberty | ✓ | ✓ |
| Roe v. Wade | Privacy Liberty | Privacy/Liberty | Privacy/Liberty | ✓ | ✓ |
| Lawrence v. Texas | Privacy Liberty | Due Process | Due Process | ✓ | ✓ |
| Dobbs v. Jackson Women's Healt | Privacy Liberty | Due Process | Privacy/Liberty | ✓ | ✓ |
| Cruzan v. Director, Missouri D | Privacy Liberty | Due Process | Privacy/Liberty | ✓ | ◐ |
| New York Times Co. v. Sullivan | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Cohen v. California | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Hustler Magazine v. Falwell | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Reno v. American Civil Liberti | Free Expression | Free Expression | Free Expression | ✓ | ✓ |
| Plessy v. Ferguson | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Craig v. Boren | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Regents of the University of C | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| United States v. Virginia | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Mapp v. Ohio | Due Process | Equal Protection | Privacy/Liberty | ✗ | ✓ |
| Rochin v. California | Due Process | Due Process | Due Process | ✓ | ✓ |
| Goldberg v. Kelly | Due Process | Due Process | Due Process | ✓ | ✓ |
| Planned Parenthood of Southeas | Due Process | Due Process | Due Process | ✓ | ✓ |
| Gibbons v. Ogden | Federalism | Federalism | Federalism | ✓ | ✓ |
| Heart of Atlanta Motel, Inc. v | Federalism | Equal Protection | Equal Protection | ✗ | ✗ |
| United States v. Morrison | Federalism | Equal Protection | Equal Protection | ◐ | ◐ |
| Gonzales v. Raich | Federalism | Federalism | Federalism | ✓ | ✓ |
| Eisenstadt v. Baird | Privacy Liberty | Equal Protection | Equal Protection | ✓ | ✓ |
| Washington v. Glucksberg | Privacy Liberty | Due Process | Due Process | ✓ | ✓ |
| Katz v. United States | Privacy Liberty | Privacy/Liberty | Privacy/Liberty | ◐ | ◐ |
| Whalen v. Roe | Privacy Liberty | Privacy/Liberty | Privacy/Liberty | ✓ | ✓ |
| Troxel v. Granville | Privacy Liberty | Due Process | Due Process | ✓ | ✓ |

---

## Overall Accuracy

**Exact matches (weight ≥ 0.8):**
- Base model: 42/49 (85.7%)
- IT model: 43/49 (87.8%)

**Partial matches (weight ≥ 0.3):**
- Base model: 4/49
- IT model: 5/49

**Weighted Score (exact=1, partial=0.5):**
- Base model: 44.0/49 (89.8%)
- IT model: 45.5/49 (92.9%)
- **Improvement**: +1.5 points (+3.1pp)

---

## Example Responses

### Case: Tinker v. Des Moines Independent Community School District

**Facts**: Public school students in Des Moines, Iowa were suspended for wearing black armbands to school to protest the Vietnam War. The students wore the armbands quietly and did not disrupt class activities o...

**Question**: Does a public school's prohibition against students wearing armbands as a form of symbolic protest violate the students' First Amendment rights to free expression?

**Ground Truth**: free_expression

#### Base Model Response:
```
1. Free Expression - The First Amendment protects the right to free expression, including symbolic speech, and the students' armbands were a form of symbolic protest.
2. Equal Protection - The school's policy banning armbands was applied equally to all students, so it does not violate the Equal Protection Clause.
3. Due Process - The students were given the opportunity to express their views and were not subject to arbitrary or capricious treatment, so their Due Process rights were not violated.
4. Federalism - The case does not involve any issues related to federalism, as it is a matter of state law.
5. Privacy/Liberty - The case does not involve any issues related to privacy or liberty, as it is a matter of free expression.
```

#### IT Model Response:
```
1. Free Expression - This case directly involves the students' right to express their views through symbolic speech, which is protected under the First Amendment.
2. Due Process - The students were disciplined without a fair process, as the school had a pre-existing policy that was applied retroactively to silence their protest.
3. Equal Protection - While not the primary issue, there is an implication that the policy may have been applied unequally if other forms of political expression were allowed but armbands were banned.
4. Privacy/Liberty - Although less directly applicable, the students' liberty to engage in peaceful protest is also at stake.
5. Federalism - Not directly relevant to this specific case, as it pertains more to the division of powers between the federal government and states, rather than the relationship between the state and its citizens in this context.
```