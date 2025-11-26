"""
SCOTUS Cases for Constitutional Geometry PoC
28 landmark cases with fact summaries and legal questions
"""

CASES = [
    # === FREE EXPRESSION CASES ===
    {
        "case_id": "tinker_1969",
        "case_name": "Tinker v. Des Moines Independent Community School District",
        "year": 1969,
        "vote": "7-2",
        "citation": "393 U.S. 503",
        "facts": """Public school students in Des Moines, Iowa were suspended for wearing 
black armbands to school to protest the Vietnam War. The students wore the armbands 
quietly and did not disrupt class activities or school operations. School officials 
had adopted a policy banning armbands specifically in anticipation of the protest, 
though other political symbols were permitted.""",
        "legal_question": """Does a public school's prohibition against students wearing 
armbands as a form of symbolic protest violate the students' First Amendment rights 
to free expression?""",
        "courtlistener_id": "tinker-v-des-moines-independent-community-school-dist",
    },
    {
        "case_id": "brandenburg_1969",
        "case_name": "Brandenburg v. Ohio",
        "year": 1969,
        "vote": "9-0",
        "citation": "395 U.S. 444",
        "facts": """A Ku Klux Klan leader was convicted under Ohio's Criminal Syndicalism 
statute for advocating violence at a televised Klan rally. At the rally, he made 
speeches suggesting possible 'revengeance' against the government and racial minorities, 
though no specific violent acts were planned or carried out.""",
        "legal_question": """Does a state law that punishes advocacy of illegal conduct 
violate the First Amendment when the speech does not incite imminent lawless action?""",
        "courtlistener_id": "brandenburg-v-ohio",
    },
    {
        "case_id": "texas_v_johnson_1989",
        "case_name": "Texas v. Johnson",
        "year": 1989,
        "vote": "5-4",
        "citation": "491 U.S. 397",
        "facts": """Gregory Johnson burned an American flag outside the Republican National 
Convention in Dallas as a form of political protest. He was convicted under a Texas 
law prohibiting desecration of a venerated object. No one was physically injured or 
threatened, though some witnesses were seriously offended.""",
        "legal_question": """Does a state law criminalizing the desecration of the American 
flag violate the First Amendment's protection of symbolic speech?""",
        "courtlistener_id": "texas-v-johnson",
    },
    {
        "case_id": "citizens_united_2010",
        "case_name": "Citizens United v. Federal Election Commission",
        "year": 2010,
        "vote": "5-4",
        "citation": "558 U.S. 310",
        "facts": """Citizens United, a nonprofit corporation, produced a documentary film 
critical of Hillary Clinton and sought to distribute it during the 2008 primary 
election season. The FEC prohibited the broadcast under the Bipartisan Campaign 
Reform Act, which barred corporations from funding electioneering communications 
within 30 days of a primary.""",
        "legal_question": """Do the provisions of the Bipartisan Campaign Reform Act that 
prohibit corporations from using general treasury funds for independent electoral 
advocacy violate the First Amendment?""",
        "courtlistener_id": "citizens-united-v-federal-election-comn",
    },
    {
        "case_id": "snyder_v_phelps_2011",
        "case_name": "Snyder v. Phelps",
        "year": 2011,
        "vote": "8-1",
        "citation": "562 U.S. 443",
        "facts": """Members of Westboro Baptist Church picketed near the funeral of Marine 
Lance Corporal Matthew Snyder, who was killed in Iraq. The picketers displayed signs 
with messages like 'God Hates Fags' and 'Thank God for Dead Soldiers' on public land 
approximately 1,000 feet from the church. The Marine's father sued for intentional 
infliction of emotional distress.""",
        "legal_question": """Does the First Amendment protect protesters who picket near 
a military funeral with offensive signs addressing matters of public concern from 
tort liability for intentional infliction of emotional distress?""",
        "courtlistener_id": "snyder-v-phelps",
    },
    {
        "case_id": "303_creative_2023",
        "case_name": "303 Creative LLC v. Elenis",
        "year": 2023,
        "vote": "6-3",
        "citation": "600 U.S. 570",
        "facts": """A website designer in Colorado wished to expand her business to include 
wedding websites but did not want to create websites celebrating same-sex marriages 
due to her religious beliefs. Colorado's Anti-Discrimination Act prohibited businesses 
offering services to the public from denying services based on sexual orientation.""",
        "legal_question": """Does requiring a website designer to create expressive content 
celebrating same-sex marriages, contrary to her sincerely held religious beliefs, 
violate her First Amendment right against compelled speech?""",
        "courtlistener_id": "303-creative-llc-v-elenis",
    },
    
    # === EQUAL PROTECTION CASES ===
    {
        "case_id": "brown_1954",
        "case_name": "Brown v. Board of Education of Topeka",
        "year": 1954,
        "vote": "9-0",
        "citation": "347 U.S. 483",
        "facts": """Black children in Topeka, Kansas were denied admission to public schools 
attended by white children under laws requiring or permitting racial segregation. 
The segregated schools were substantially equal in terms of buildings, curricula, 
qualifications of teachers, and other tangible factors. The plaintiffs challenged 
the 'separate but equal' doctrine established in Plessy v. Ferguson.""",
        "legal_question": """Does racial segregation of children in public schools, even 
where physical facilities and other tangible factors are equal, deprive minority 
children of equal protection under the Fourteenth Amendment?""",
        "courtlistener_id": "brown-v-board-of-education",
    },
    {
        "case_id": "loving_1967",
        "case_name": "Loving v. Virginia",
        "year": 1967,
        "vote": "9-0",
        "citation": "388 U.S. 1",
        "facts": """Mildred Jeter, a Black woman, and Richard Loving, a white man, were 
married in Washington, D.C. and returned to Virginia, where they were charged with 
violating Virginia's law banning interracial marriages. They pleaded guilty and 
were sentenced to one year in prison, suspended on condition they leave Virginia 
for 25 years.""",
        "legal_question": """Does a state law prohibiting interracial marriage violate 
the Equal Protection and Due Process Clauses of the Fourteenth Amendment?""",
        "courtlistener_id": "loving-v-virginia",
    },
    {
        "case_id": "reed_1971",
        "case_name": "Reed v. Reed",
        "year": 1971,
        "vote": "7-0",
        "citation": "404 U.S. 71",
        "facts": """After the death of their adopted son, Sally and Cecil Reed, who were 
separated, both filed petitions to be appointed administrator of their son's estate. 
An Idaho statute provided that 'males must be preferred to females' when multiple 
persons in the same entitlement class seek appointment as administrator.""",
        "legal_question": """Does an Idaho statute that mandates preference for males 
over females in appointing administrators of estates violate the Equal Protection 
Clause of the Fourteenth Amendment?""",
        "courtlistener_id": "reed-v-reed",
    },
    {
        "case_id": "grutter_2003",
        "case_name": "Grutter v. Bollinger",
        "year": 2003,
        "vote": "5-4",
        "citation": "539 U.S. 306",
        "facts": """Barbara Grutter, a white Michigan resident with a 3.8 GPA and 161 LSAT 
score, was denied admission to the University of Michigan Law School. The Law School's 
admissions policy considered race as one factor among many to achieve student body 
diversity, giving substantial weight to diversity considerations but without using 
quotas or mechanical formulas.""",
        "legal_question": """Does a public university's use of race as a factor in law 
school admissions, to achieve educational benefits of diversity without quotas, 
violate the Equal Protection Clause of the Fourteenth Amendment?""",
        "courtlistener_id": "grutter-v-bollinger",
    },
    {
        "case_id": "obergefell_2015",
        "case_name": "Obergefell v. Hodges",
        "year": 2015,
        "vote": "5-4",
        "citation": "576 U.S. 644",
        "facts": """Same-sex couples in Ohio, Michigan, Kentucky, and Tennessee challenged 
state laws defining marriage as between a man and a woman, and refusing to recognize 
same-sex marriages lawfully performed in other states. The plaintiffs sought the 
right to marry and to have their existing marriages recognized.""",
        "legal_question": """Does the Fourteenth Amendment require states to license 
marriages between same-sex couples and to recognize same-sex marriages lawfully 
performed in other states?""",
        "courtlistener_id": "obergefell-v-hodges",
    },
    {
        "case_id": "sffa_harvard_2023",
        "case_name": "Students for Fair Admissions v. President and Fellows of Harvard College",
        "year": 2023,
        "vote": "6-3",
        "citation": "600 U.S. 181",
        "facts": """Students for Fair Admissions challenged admissions programs at Harvard 
and the University of North Carolina that considered race as a factor in admissions 
decisions. The programs aimed to achieve diversity but were alleged to discriminate 
against Asian American applicants and to use race as more than a 'plus factor.'""",
        "legal_question": """Do race-conscious admissions programs at Harvard and UNC 
violate the Equal Protection Clause of the Fourteenth Amendment?""",
        "courtlistener_id": "students-for-fair-admissions-inc-v-president-fellows-of-harvard-college",
    },
    
    # === DUE PROCESS CASES ===
    {
        "case_id": "gideon_1963",
        "case_name": "Gideon v. Wainwright",
        "year": 1963,
        "vote": "9-0",
        "citation": "372 U.S. 335",
        "facts": """Clarence Earl Gideon was charged with breaking and entering in Florida, 
a felony under state law. Being unable to afford an attorney, he requested that 
the court appoint one for him. The Florida court denied his request, stating that 
under Florida law, counsel could only be appointed for defendants charged with 
capital offenses. Gideon represented himself and was convicted.""",
        "legal_question": """Does the Sixth Amendment's guarantee of the right to counsel, 
as applied to the states through the Fourteenth Amendment, require states to provide 
attorneys to criminal defendants who cannot afford them?""",
        "courtlistener_id": "gideon-v-wainwright",
    },
    {
        "case_id": "miranda_1966",
        "case_name": "Miranda v. Arizona",
        "year": 1966,
        "vote": "5-4",
        "citation": "384 U.S. 436",
        "facts": """Ernesto Miranda was arrested for kidnapping and rape. After two hours 
of police interrogation without being informed of his right to an attorney or his 
right to remain silent, Miranda signed a confession. The confession was admitted 
at trial, and Miranda was convicted. He had not been advised that anything he said 
could be used against him.""",
        "legal_question": """Does the Fifth Amendment's protection against self-incrimination 
require police to inform suspects of their rights before custodial interrogation?""",
        "courtlistener_id": "miranda-v-arizona",
    },
    {
        "case_id": "mathews_1976",
        "case_name": "Mathews v. Eldridge",
        "year": 1976,
        "vote": "6-2",
        "citation": "424 U.S. 319",
        "facts": """George Eldridge's Social Security disability benefits were terminated 
based on reports from his physician and a state agency, without a hearing before 
the termination. He was advised of the proposed termination and given an opportunity 
to submit written evidence, but was not offered an evidentiary hearing until after 
benefits had been cut off.""",
        "legal_question": """Does due process require an evidentiary hearing before 
termination of Social Security disability benefits, or is a post-termination 
hearing sufficient?""",
        "courtlistener_id": "mathews-v-eldridge",
    },
    {
        "case_id": "bmw_v_gore_1996",
        "case_name": "BMW of North America, Inc. v. Gore",
        "year": 1996,
        "vote": "5-4",
        "citation": "517 U.S. 559",
        "facts": """Dr. Ira Gore purchased a new BMW from an authorized dealer and later 
discovered it had been repainted due to acid rain damage during transport. BMW had 
a nationwide policy of not disclosing repairs costing less than 3% of the car's 
value. An Alabama jury awarded $4,000 in compensatory damages and $4 million in 
punitive damages, later reduced to $2 million.""",
        "legal_question": """Does an award of punitive damages grossly exceeding 
compensatory damages violate the Due Process Clause of the Fourteenth Amendment?""",
        "courtlistener_id": "bmw-of-north-america-inc-v-gore",
    },
    {
        "case_id": "hamdi_2004",
        "case_name": "Hamdi v. Rumsfeld",
        "year": 2004,
        "vote": "6-3",
        "citation": "542 U.S. 507",
        "facts": """Yaser Hamdi, an American citizen, was captured in Afghanistan during 
the 2001 conflict and designated an 'enemy combatant' by the U.S. government. He 
was held without charges in military custody with no access to counsel and no 
opportunity to challenge his detention before a neutral decision-maker.""",
        "legal_question": """Does due process require that a U.S. citizen held as an 
enemy combatant be given a meaningful opportunity to contest the factual basis 
for his detention before a neutral decision-maker?""",
        "courtlistener_id": "hamdi-v-rumsfeld",
    },
    {
        "case_id": "trump_v_hawaii_2018",
        "case_name": "Trump v. Hawaii",
        "year": 2018,
        "vote": "5-4",
        "citation": "138 S. Ct. 2392",
        "facts": """President Trump issued a proclamation restricting entry into the United 
States of nationals from several predominantly Muslim countries, citing national 
security concerns. Challengers argued the proclamation was motivated by religious 
animus, pointing to the President's campaign statements about banning Muslims.""",
        "legal_question": """Does a presidential proclamation restricting entry from 
specified countries violate the Establishment Clause or exceed the President's 
authority under the Immigration and Nationality Act?""",
        "courtlistener_id": "trump-v-hawaii",
    },
    
    # === FEDERALISM CASES ===
    {
        "case_id": "mcculloch_1819",
        "case_name": "McCulloch v. Maryland",
        "year": 1819,
        "vote": "9-0",
        "citation": "17 U.S. 316",
        "facts": """Congress chartered the Second Bank of the United States. The state of 
Maryland enacted a law imposing a tax on all banks not chartered by the state, 
effectively targeting only the Bank of the United States. James McCulloch, a 
cashier at the Baltimore branch, refused to pay the tax.""",
        "legal_question": """Does Congress have constitutional authority to charter a 
national bank, and can a state tax an instrumentality of the federal government?""",
        "courtlistener_id": "mcculloch-v-state-of-maryland",
    },
    {
        "case_id": "lopez_1995",
        "case_name": "United States v. Lopez",
        "year": 1995,
        "vote": "5-4",
        "citation": "514 U.S. 549",
        "facts": """Alfonso Lopez, a high school senior in San Antonio, Texas, carried a 
concealed handgun into his school. He was charged under the federal Gun-Free School 
Zones Act of 1990, which made it a federal offense to knowingly possess a firearm 
in a school zone. The government argued Congress had power to regulate under the 
Commerce Clause.""",
        "legal_question": """Does the Gun-Free School Zones Act exceed Congress's power 
under the Commerce Clause because possession of a firearm in a school zone does 
not substantially affect interstate commerce?""",
        "courtlistener_id": "united-states-v-lopez",
    },
    {
        "case_id": "printz_1997",
        "case_name": "Printz v. United States",
        "year": 1997,
        "vote": "5-4",
        "citation": "521 U.S. 898",
        "facts": """The Brady Handgun Violence Prevention Act required state and local 
law enforcement officers to conduct background checks on prospective handgun 
purchasers. County sheriffs challenged this requirement, arguing that Congress 
could not compel state officers to execute federal law.""",
        "legal_question": """Does the Brady Act's requirement that state law enforcement 
officers conduct background checks on handgun purchasers violate principles of 
state sovereignty by commandeering state officials?""",
        "courtlistener_id": "printz-v-united-states",
    },
    {
        "case_id": "nfib_v_sebelius_2012",
        "case_name": "National Federation of Independent Business v. Sebelius",
        "year": 2012,
        "vote": "5-4",
        "citation": "567 U.S. 519",
        "facts": """The Affordable Care Act required most Americans to obtain health 
insurance or pay a penalty (the 'individual mandate') and expanded Medicaid 
eligibility while threatening states with loss of all Medicaid funding if they 
refused to expand. Multiple states and individuals challenged these provisions.""",
        "legal_question": """Does Congress have power under the Commerce Clause or Taxing 
Power to require individuals to purchase health insurance, and can Congress 
condition all Medicaid funding on state acceptance of expanded coverage?""",
        "courtlistener_id": "national-federation-of-independent-business-v-sebelius",
    },
    {
        "case_id": "murphy_2018",
        "case_name": "Murphy v. National Collegiate Athletic Association",
        "year": 2018,
        "vote": "6-3",
        "citation": "138 S. Ct. 1461",
        "facts": """The Professional and Amateur Sports Protection Act prohibited states 
from authorizing sports gambling. New Jersey sought to legalize sports betting 
and challenged PASPA as unconstitutional. The federal government argued PASPA 
merely preempted contrary state law rather than commanding state action.""",
        "legal_question": """Does a federal law that prohibits states from authorizing 
sports gambling violate the anti-commandeering doctrine, which prevents Congress 
from commanding states to enact or enforce a federal regulatory program?""",
        "courtlistener_id": "murphy-v-national-collegiate-athletic-association",
    },
    
    # === PRIVACY/LIBERTY CASES ===
    {
        "case_id": "griswold_1965",
        "case_name": "Griswold v. Connecticut",
        "year": 1965,
        "vote": "7-2",
        "citation": "381 U.S. 479",
        "facts": """Estelle Griswold, executive director of Planned Parenthood League of 
Connecticut, was convicted under a state law that criminalized providing counseling 
and medical treatment to married persons for purposes of preventing conception. 
She had provided information, instruction, and medical advice to married couples 
about contraception.""",
        "legal_question": """Does a state law criminalizing the use of contraceptives 
by married couples violate a constitutional right to privacy?""",
        "courtlistener_id": "griswold-v-connecticut",
    },
    {
        "case_id": "roe_1973",
        "case_name": "Roe v. Wade",
        "year": 1973,
        "vote": "7-2",
        "citation": "410 U.S. 113",
        "facts": """Jane Roe, an unmarried pregnant woman, challenged a Texas law that 
criminalized abortion except to save the life of the mother. She claimed the law 
violated her constitutional right to privacy. Texas argued it had an interest in 
protecting prenatal life from the moment of conception.""",
        "legal_question": """Does the Constitution recognize a woman's right to terminate 
her pregnancy, and if so, what state interests justify limiting that right?""",
        "courtlistener_id": "roe-v-wade",
    },
    {
        "case_id": "lawrence_2003",
        "case_name": "Lawrence v. Texas",
        "year": 2003,
        "vote": "6-3",
        "citation": "539 U.S. 558",
        "facts": """Police responding to a reported weapons disturbance entered John 
Lawrence's apartment and observed him engaging in a sexual act with another man. 
Both men were arrested and convicted under a Texas statute criminalizing 'deviate 
sexual intercourse' between persons of the same sex.""",
        "legal_question": """Does a Texas statute criminalizing private, consensual 
sexual conduct between adults of the same sex violate the Due Process Clause 
of the Fourteenth Amendment?""",
        "courtlistener_id": "lawrence-v-texas",
    },
    {
        "case_id": "dobbs_2022",
        "case_name": "Dobbs v. Jackson Women's Health Organization",
        "year": 2022,
        "vote": "6-3",
        "citation": "597 U.S. 215",
        "facts": """Mississippi enacted the Gestational Age Act, which prohibited abortion 
after 15 weeks of pregnancy except in medical emergencies or severe fetal abnormality. 
Jackson Women's Health Organization, the state's only licensed abortion facility, 
challenged the law as unconstitutional under Roe v. Wade and Planned Parenthood v. 
Casey.""",
        "legal_question": """Should the Court overrule Roe v. Wade and Planned Parenthood 
v. Casey, and are all pre-viability prohibitions on elective abortions 
unconstitutional?""",
        "courtlistener_id": "dobbs-v-jackson-womens-health-organization",
    },
    {
        "case_id": "cruzan_1990",
        "case_name": "Cruzan v. Director, Missouri Department of Health",
        "year": 1990,
        "vote": "5-4",
        "citation": "497 U.S. 261",
        "facts": """Nancy Cruzan was in a persistent vegetative state following an 
automobile accident. Her parents sought to remove her feeding tube, but Missouri 
required clear and convincing evidence of the patient's wishes before allowing 
withdrawal of life-sustaining treatment. No such clear evidence of Nancy's 
wishes existed.""",
        "legal_question": """Does the Constitution grant a competent person a protected 
liberty interest in refusing unwanted medical treatment, and may a state require 
clear and convincing evidence of an incompetent patient's wishes?""",
        "courtlistener_id": "cruzan-v-director-mo-dept-of-health",
    },
]

# Convenience accessors
def get_case(case_id: str) -> dict:
    """Get a case by its ID."""
    for case in CASES:
        if case["case_id"] == case_id:
            return case
    raise ValueError(f"Case not found: {case_id}")

def get_all_case_ids() -> list:
    """Get all case IDs."""
    return [case["case_id"] for case in CASES]

def format_prompt(case: dict) -> str:
    """Format a case into the standard prompt for activation extraction."""
    return f"""Consider the following case before the Supreme Court:

FACTS: {case['facts'].strip()}

LEGAL QUESTION: {case['legal_question'].strip()}

The relevant constitutional principles that may apply include:
- Free Expression (First Amendment speech, press, association)
- Equal Protection (Fourteenth Amendment equal treatment)
- Due Process (Fifth/Fourteenth Amendment procedural and substantive)
- Federalism (state vs. federal power, Tenth Amendment)
- Privacy and Liberty (unenumerated rights, substantive due process liberty interest)

How should the Court rule, and which principles should guide the decision?"""
