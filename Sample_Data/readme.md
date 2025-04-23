The Sample_Data folder contains the sample of NeGoChat and IND datasets.

> sample_NeGoChat dataset contains 5 columns:

	> conversation_id: Unique conversation id

	> turn_no: Utterance number in an ongoing conversation	

	> speaker: Can take either of the two roles: Travel Agent/Traveler.

	> utterance: The text stated by Travel Agent/Traveler.	

	> negotiation_strategy: Negotiation strategy employed by the travel agent and traveler. It can be one of the 8 strategies: 'active_listening', 'leverage_information', 'no_strategy', 'expanding_the_pie', 'gradual_concession_making', 'large_initial_concession_making', 'logrolling', 'patterned_concession_making'


> sample_IND dataset contains 9 columns:

	> conversation_id: Unique conversation id

	> turn_no: Utterance number in an ongoing conversation	

	> speaker: Can take either of the two roles: Buyer/Seller.
		>> 0 - buyer
		>> 1 - seller

	> utterance: The text stated by Travel Agent/Traveler.	

	> intent: Intent of the speaker. 

	> price: Price at current turn

	> background_data: Background information associated with the electronic items or its associated accessories
	
	> items: Electronic item and its associated accessories

	> negotiation_strategy: Negotiation strategy employed by the travel agent and traveler. It can be one of the 8 strategies: 'active_listening', 'leverage_information', 'no_strategy', 'expanding_the_pie', 'gradual_concession_making', 'large_initial_concession_making', 'logrolling', 'patterned_concession_making'
	