# acl2018-mds.p is the goal_set.p used in code.

Loading this dataset using the following command in Python:
    import pickle
    data_set = pickle.load(open(file_name, 'rb'))

1. acl2018-mds.p: the goal_set.p used in the code. The goal_set contains training set and testing set, which can be visited with goal_set["train"] and goal_set["test"]. Each sub-set is a list of user goals expalined in our paper and each user goal is an dictionary which has three keys, "consult_id" is the user id, "disease_tag" is the disease that the user suffers and  "goal" is the combination of slots (request slots, implicit symptoms and explicit symptoms).

2. action_set.p: the types of action pre-defined for this medical DS.

3. disease_symptom.p: the collection of symptoms for each disease.

4. slot_set.p: the set of slots, which consists of normalized symptoms and a special slot diseaseas as explained in our paper.

Please see our paper for details.