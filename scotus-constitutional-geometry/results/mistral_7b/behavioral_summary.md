# Behavioral Verification Summary

**Model Pair**: mistral-7b
- Base: mistralai/Mistral-7B-v0.1
- Aligned: mistralai/Mistral-7B-Instruct-v0.1

**Cases Evaluated**: 49
**Timestamp**: 2025-12-21T05:02:38.285239

---

## Principle Identification Accuracy

*Scoring: ✓ = exact match (weight ≥ 0.8), ◐ = partial match (weight ≥ 0.3), ✗ = miss (weight < 0.3)*

| Case | Primary | Base Pick | Aligned Pick | Base | Aligned |
|------|---------|-----------|--------------|------|---------|
| Tinker v. Des Moines Independe | Free Expression | - | Free Expression | ✗ | ✓ |
| Brandenburg v. Ohio | Free Expression | - | Free Expression | ✗ | ✓ |
| Texas v. Johnson | Free Expression | - | Free Expression | ✗ | ✓ |
| Citizens United v. Federal Ele | Free Expression | - | Free Expression | ✗ | ✓ |
| Snyder v. Phelps | Free Expression | - | Free Expression | ✗ | ✓ |
| 303 Creative LLC v. Elenis | Free Expression | - | Free Expression | ✗ | ✓ |
| Brown v. Board of Education of | Equal Protection | Free Expression | Equal Protection | ✗ | ✓ |
| Loving v. Virginia | Equal Protection | - | Equal Protection | ✗ | ✓ |
| Reed v. Reed | Equal Protection | - | Equal Protection | ✗ | ✓ |
| Grutter v. Bollinger | Equal Protection | - | Equal Protection | ✗ | ✓ |
| Obergefell v. Hodges | Equal Protection | - | Equal Protection | ✗ | ◐ |
| Students for Fair Admissions v | Equal Protection | - | Equal Protection | ✗ | ✓ |
| Gideon v. Wainwright | Due Process | - | Due Process | ✗ | ✓ |
| Miranda v. Arizona | Due Process | - | Due Process | ✗ | ✓ |
| Mathews v. Eldridge | Due Process | - | Due Process | ✗ | ✓ |
| BMW of North America, Inc. v.  | Due Process | - | Due Process | ✗ | ✓ |
| Hamdi v. Rumsfeld | Due Process | Due Process | Due Process | ✓ | ✓ |
| Trump v. Hawaii | Due Process | - | Equal Protection | ✗ | ◐ |
| McCulloch v. Maryland | Federalism | - | Federalism | ✗ | ✓ |
| United States v. Lopez | Federalism | Free Expression | Due Process | ✗ | ✗ |
| Printz v. United States | Federalism | Federalism | Federalism | ✓ | ✓ |
| National Federation of Indepen | Federalism | Federalism | Federalism | ✓ | ✓ |
| Murphy v. National Collegiate  | Federalism | Free Expression | Federalism | ✗ | ✓ |
| Griswold v. Connecticut | Privacy Liberty | - | Privacy/Liberty | ✗ | ✓ |
| Roe v. Wade | Privacy Liberty | - | Privacy/Liberty | ✗ | ✓ |
| Lawrence v. Texas | Privacy Liberty | Due Process | Privacy/Liberty | ✓ | ✓ |
| Dobbs v. Jackson Women's Healt | Privacy Liberty | - | Privacy/Liberty | ✗ | ✓ |
| Cruzan v. Director, Missouri D | Privacy Liberty | - | Due Process | ✗ | ✓ |
| New York Times Co. v. Sullivan | Free Expression | - | Free Expression | ✗ | ✓ |
| Cohen v. California | Free Expression | - | Free Expression | ✗ | ✓ |
| Hustler Magazine v. Falwell | Free Expression | - | Free Expression | ✗ | ✓ |
| Reno v. American Civil Liberti | Free Expression | - | Free Expression | ✗ | ✓ |
| Plessy v. Ferguson | Equal Protection | Equal Protection | Equal Protection | ✓ | ✓ |
| Craig v. Boren | Equal Protection | Free Expression | Equal Protection | ✗ | ✓ |
| Regents of the University of C | Equal Protection | - | Equal Protection | ✗ | ✓ |
| United States v. Virginia | Equal Protection | - | Equal Protection | ✗ | ✓ |
| Mapp v. Ohio | Due Process | - | Privacy/Liberty | ✗ | ✓ |
| Rochin v. California | Due Process | Due Process | Due Process | ✓ | ✓ |
| Goldberg v. Kelly | Due Process | Free Expression | Due Process | ✗ | ✓ |
| Planned Parenthood of Southeas | Due Process | - | Due Process | ✗ | ✓ |
| Gibbons v. Ogden | Federalism | - | Federalism | ✗ | ✓ |
| Heart of Atlanta Motel, Inc. v | Federalism | - | Equal Protection | ✗ | ✗ |
| United States v. Morrison | Federalism | - | Equal Protection | ✗ | ◐ |
| Gonzales v. Raich | Federalism | - | Federalism | ✗ | ✓ |
| Eisenstadt v. Baird | Privacy Liberty | - | Equal Protection | ✗ | ✓ |
| Washington v. Glucksberg | Privacy Liberty | - | Due Process | ✗ | ✓ |
| Katz v. United States | Privacy Liberty | Free Expression | Privacy/Liberty | ✗ | ◐ |
| Whalen v. Roe | Privacy Liberty | - | Privacy/Liberty | ✗ | ✓ |
| Troxel v. Granville | Privacy Liberty | Due Process | Due Process | ✓ | ✓ |

---

## Overall Accuracy

**Exact matches (weight ≥ 0.8):**
- Base model: 7/49 (14.3%)
- Aligned model: 43/49 (87.8%)

**Partial matches (weight ≥ 0.3):**
- Base model: 0/49
- Aligned model: 4/49

**Weighted Score (exact=1, partial=0.5):**
- Base model: 7.0/49 (14.3%)
- Aligned model: 45.0/49 (91.8%)
- **Improvement**: +38.0 points (+77.6pp)

---

## Example Responses

### Case: Tinker v. Des Moines Independent Community School District

**Facts**: Public school students in Des Moines, Iowa were suspended for wearing black armbands to school to protest the Vietnam War. The students wore the armbands quietly and did not disrupt class activities o...

**Question**: Does a public school's prohibition against students wearing armbands as a form of symbolic protest violate the students' First Amendment rights to free expression?

**Ground Truth**: free_expression

#### Base Model Response:
```
1. [Principle]
2. [Principle]
3. [Principle]
4. [Principle]
5. [Principle]

Explain your ranking.

1. [Principle]

2. [Principle]

3. [Principle]

4. [Principle]

5. [Principle]

© BrainMass Inc. brainmass.com October 10, 2019, 10:29 am ad1c9bdddf
https://brainmass.com/law/constitutional-law/constitutional-principles-tinker-v-des-moines-independent-community-school-district-444485

This solution discusses the constitutional principles that apply to the case of Tinker v.
```

#### Aligned Model Response:
```
1. Free Expression - This principle is the most relevant to this case as it protects the right of individuals to express their ideas and beliefs without fear of government censorship or retaliation.
2. Equal Protection - The principle of equal protection requires that all individuals be treated equally under the law, regardless of their political beliefs or actions. In this case, the school's policy banning armbands specifically for political protest violates this principle by treating students who wish to express their opposition to the Vietnam War differently than those who wish to express their support for it.
3. Due Process - The principle of due process requires that individuals be given fair notice of any actions that will be taken against them and an opportunity to be heard. In this case, the school's policy banning armbands was adopted in anticipation of the protest, but the students were not given any notice or opportunity to be heard before being suspended.
4. Federalism - This principle
```