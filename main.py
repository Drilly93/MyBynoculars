from binoculars import *

HumanText = '''They were facing the president's promises to unseat them in primaries, pressure from Vice President JD Vance and House Speaker Mike Johnson and a large number of physical threats. (Law enforcement officials have not linked the threats to any group or campaign.)
In other words, these Republicans would have known precisely the potentially severe costs of their votes — and a majority of them still voted against Trump.
The vote was also significant in another way: It might have put a nail in the coffin of Trump’s big redistricting push. Without gaining two favorable districts in Indiana (as the map proposed), Trump’s bare-knuckle push for states to gerrymander in the middle of the decade to help the GOP next year looks to be fizzling.
Republicans might gain an advantage in a handful of seats, but it's looking more and more like it will be pretty close to a wash.'''

AIText = '''France's economy is one of the largest in Europe and plays a central role within the European Union. It is characterized by a strong mix of public and private sectors, with the government traditionally taking an active role in strategic industries such as energy, transport, and defense. France benefits from a highly diversified economy, including manufacturing, agriculture, tourism, and a large services sector that represents the majority of GDP.In recent years, the French economy has faced several challenges, including slower growth, high public debt, and inflationary pressures driven by global energy prices. At the same time, France has invested heavily in innovation, digitalization, and green technologies to improve productivity and long-term competitiveness.'''


# Code execution
bino = Binoculars()

print("Human text Score :", bino.compute_score(HumanText))  
print(bino.predict(HumanText))  

print("AI text Score :", bino.compute_score(AIText))
print(bino.predict(AIText))  